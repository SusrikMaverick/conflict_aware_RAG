"""Our pipeline: answer a question from documents that disagree (design: plan/plan.md).

It keeps two jobs apart. What kind of disagreement is this? That decides the form of the
answer. Who is right? That is asked only for real contradictions, and answered only from
evidence in the documents.

    1 intake      note whether the question names a time, and whether its answer can change
    2 retrieve    the documents given with the question are the collection
    3 extract     one document at a time -> fact cards
    4 verify      keep a card only if its quote is really in its document
    5 map         group the cards by entity -> sense -> answer; count independent sources
    6 explain     look for a quoted explanation before calling anything a contradiction
    7 label       each pair of answers gets one of seven relations
    8 adjudicate  contradictions only: source information, outside facts, majority, memory
    9 answer      entries built from the labels, each with its sources and basis

`mode` says what happens when step 8 finds nothing that settles a contradiction:
"disagree" reports both sides; "forced" picks the better-supported side and says so.

    from src.pipeline import run
    result = run(llm, question, [{"text": ...}, ...], mode="forced")
    result["response"], result["entries"], result["status"], result["trace"]

Not built yet: searching a large collection (step 2), MiniCheck's check that a document
supports a card (step 4), and the trained relation classifier (step 7 prompts the model).

Try it:  python -m src.pipeline [question.json] [--mode forced] [--memory] [--trace]
"""
import argparse
import calendar
import collections
import itertools
import json
import re
import string
import unicodedata
from difflib import SequenceMatcher

MODES = ["disagree", "forced"]
RELATIONS = ["agree", "different entity", "different sense", "different time",
             "complementary", "opinion", "contradiction"]  # from least to most serious
FIELDS = ["entity", "sense", "answer", "as_of", "claim_kind", "attributed_to", "evidence", "claim"]  # a fact card
DECODING = {"temperature": 0, "extra_body": {"repetition_penalty": 1.05}}  # greedy, as for every method we compare

# First values, not tuned yet. They are to be set on practice data, never on a test set.
CLOSE = 0.8        # step 4: share of a quote's words that must match the document
COPY = 0.8         # step 5: text similarity from which two documents count as one source
MAJORITY = True    # step 8: may a clear majority of independent sources decide? (assumes fakes are a minority)
CLEARLY_AHEAD = 2  # step 8: "clear" is this many times the runner-up's independent sources
CONFIDENT = 90     # step 8: confidence, 0 to 100, that the model's memory needs to decide

INTAKE = """\
Read this question. Do not answer it.

Question: {question}

Reply with JSON only, in this form:
{{"time": null, "changes_over_time": false}}

"time": the date or period the question itself names, such as "2015" or "March 2020"; null if it names none.
"changes_over_time": true if the right answer can change as time passes (a count, a record, who holds an office now); false if it is fixed (when a past event happened, who wrote a book)."""

RECALL = """\
Answer this question from what you already know. No documents are given.

Question: {question}

Reply with JSON only, in this form:
{{"answers": [{{"about": "", "answer": "", "confidence": 0}}]}}

Give one item for each thing the question could be about, since several things can share a name. "about" says which one. "answer" is short: a name, a date, a number or a phrase. "confidence" is how sure you are that the answer is right, from 0 to 100. If you do not know, reply {{"answers": []}}."""

EXTRACT = """\
Read one document and report what it says in answer to a question. Use only this document: nothing you know from elsewhere, and no guesses.

Question: {question}

Document:
{document}

Reply with JSON only, in this form:
{{"relevant": true, "cards": [{{"entity": "", "sense": "", "answer": "", "as_of": "", "claim_kind": "", "attributed_to": "", "evidence": "", "claim": ""}}]}}

"relevant": false if the document is about some other topic and is of no use for the question; then "cards" is [].
"cards": one card for each answer the document itself gives to the question. Usually that is one card. It is several cards if the document gives several answers, such as a list of names: one card each. It is [] if the document gives no answer. The question may fit several things that share a name: whichever of them this document is about, report its answer. If the document only says that some answer is wrong, write no card for that answer.

The fields of a card:
- "entity": who or what this answer is about, as specifically as the document identifies it: the full name, plus the details the document gives that tell it apart from others with the same name (what kind of thing it is, profession or role, nationality or place, years, creator, other names), even where the question does not ask about them. At most about 25 words.
- "sense": which event, edition, version, condition, place or measure the answer refers to, if the document says; otherwise "not stated".
- "answer": the answer, copied from the document as it is written there, and short: a full name, a date, a number with its unit, or a short phrase. For a yes-or-no question, "yes" or "no".
- "as_of": the date at which this was true, if the document says; otherwise "not stated".
- "claim_kind": "factual", or "opinion" for a judgment, a recommendation or a contested finding.
- "attributed_to": who or what the document says this comes from, such as a census, a study or a named person, if it says; otherwise "not stated".
- "evidence": the sentence of the document that gives the answer, copied word for word. Two sentences if one is not enough.
- "claim": one sentence that states the answer on its own and names the entity.

Inside a JSON string, write a double quote as \\"."""

MAP = """\
Each card below was written from one document, in answer to the same question. Sort the cards into groups.

Question: {question}

{cards}

Sort in three layers:
1. Entity. Cards about the same real-world thing go together. A shared name is not enough, because different people, places and works can have the same name: go by the identifying details (what kind of thing, role, place, years, other names). Cards that describe one thing in different words go together. Cards with the same identifying details go together even when their answers differ.
2. Sense. Inside an entity, cards about the same event, edition, version, condition, place or measure go together. Cards whose sense is "not stated" go in a sense named "not stated".
3. Answer. Inside a sense, cards with compatible answers go together: the same answer in other words or at another precision, such as "December 2009" and "18 December 2009". Answers that differ go in separate groups, whichever you believe is right.

Reply with JSON only, in this form, where each list of numbers is one group of cards with compatible answers:
{{"entities": [{{"name": "", "senses": [{{"name": "", "answers": [[1, 2], [3]]}}]}}], "uncertain": []}}

"name" of an entity: its full name with the details that identify it.
"uncertain": the numbers of the cards that could belong to more than one entity.
Every card number appears exactly once under "answers"."""

EXPLAIN = """\
Documents give different answers about the same thing. Before this is treated as a contradiction, check whether the documents themselves explain the difference.

Question: {question}

{answers}

Documents:
{documents}

Look for a sentence in the documents which shows that the answers do not conflict, because
- they refer to different senses: different events, editions, versions, conditions, places or measures (an original release and a re-release, for example), or
- they refer to different times: the value changed, and each answer was right at its own date.
Use only what the documents say: nothing you know from elsewhere, and no explanation of your own. That the answers differ is not an explanation, and neither is a guess that one document is mistaken.

Reply with JSON only. If a sentence explains the difference:
{{"explained": true, "by": "sense", "document": 1, "quote": "", "labels": {{"A": "", "B": ""}}}}
"by" is "sense" or "time". "quote" is that sentence, copied word for word, and "document" is the number of the document it is in. "labels" gives, for each answer, the sense or the date it refers to.
If no sentence explains it:
{{"explained": false}}

Inside a JSON string, write a double quote as \\"."""

LABEL = """\
Two groups of documents answer the same question. Say how the two answers relate. Do not say which one is right.

Question: {question}

{answers}

Explanation found in the documents: {explanation}

The relations:
- "agree": the same answer, possibly at another precision or in other words.
- "different entity": the answers are about different things that share a name.
- "different sense": the same entity, but a different event, edition, version, condition, place or measure.
- "different time": the same entity and sense; the value changed, and each answer was right at its own date.
- "complementary": both are true, and together they form a fuller answer.
- "opinion": judgments, recommendations or contested findings, on which sources may fairly differ.
- "contradiction": the same entity, sense and time; both cannot be true, and nothing explains the difference.

Go only by what is shown for each answer. Choose "different entity", "different sense" or "different time" only when the identifying details, senses or dates shown really differ, or the explanation shows it; do not assume a difference that is not stated. If the answers are about the same thing in the same sense and cannot both be true, it is a contradiction.

Reply with JSON only, in this form:
{{"reason": "one sentence", "relation": ""}}"""

SOURCES = """\
Documents give contradicting answers about the same thing. Decide whether what is known about the sources settles it.

Question: {question}

{answers}

What is known about each document's source:
{sources}

An official or primary source outweighs a blog, a forum post or another secondary source. A later revision of a source outweighs an earlier one. Go only by the source information shown, not by which answer you believe. If it does not clearly favor one answer, "favors" is null.

Reply with JSON only, in this form:
{{"reason": "one sentence", "favors": "A"}}"""

FACTS = """\
Documents give contradicting answers about the same thing. Other documents, which take no side, may state a fact that makes one of the answers impossible.

Question: {question}

{answers}

The other documents:
{documents}

Look in the other documents for a fact that rules one answer out: a date of birth that makes a person too young, for example, or a plain statement that the answer is false. Use only what these documents say, nothing you know from elsewhere. A fact that only makes an answer less likely does not count.

Reply with JSON only. If such a fact is there:
{{"reason": "one sentence", "impossible": "A", "document": 1, "quote": ""}}
"impossible" is the letter of the answer that is ruled out. "quote" is the sentence stating the fact, copied word for word, and "document" is the number of the document it is in.
If there is none:
{{"impossible": null}}

Inside a JSON string, write a double quote as \\"."""

CHOOSE = """\
Documents give contradicting answers about the same thing, and one answer has to be chosen. Choose the answer the documents support better.

Question: {question}

{answers}

The documents:
{documents}

Weigh the evidence. Which documents are specific, and consistent within themselves: do their names, dates and other details fit together? Which fit what the other documents say? Does any document read as altered, or out of place? {knowledge}

Reply with JSON only, in this form:
{{"reason": "one or two sentences", "choice": "A"}}"""

STEPS = {INTAKE: "intake", RECALL: "recall", EXTRACT: "extract", MAP: "map", EXPLAIN: "explain",
         LABEL: "label", SOURCES: "sources", FACTS: "facts", CHOOSE: "choose"}  # each prompt's step, for the call log

# The worked example of plan/plan.md, section 6.
EXAMPLE = {
    "question": "When was Avatar released?",
    "documents": [
        {"text": "Avatar is a 2009 epic science fiction film directed by James Cameron. It was released in "
                 "theaters in the United States on 18 December 2009."},
        {"text": "Avatar: The Way of Water is the 2022 sequel to James Cameron's Avatar. It was released in "
                 "theaters on 16 December 2022."},
        {"text": "Avatar: The Last Airbender is an American animated television series. It premiered on "
                 "Nickelodeon on 21 February 2005."},
        {"text": "James Cameron's Avatar was released in 2010."},
        {"text": "James Cameron's science fiction film Avatar opened in cinemas in December 2009 and became "
                 "the highest-grossing film of all time."},
        {"text": "In video games, an avatar is the character or icon that represents the player. Many online "
                 "games let players change how their avatar looks."},
    ],
}


# ---- Text helpers ----

def plain(text):
    """Text for matching: lowercase, no punctuation, single spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())


def stated(value):
    """False for the ways a field says nothing: "not stated", null, empty."""
    return plain(value) not in ("", "not stated", "none", "null", "unknown", "n a")


def compatible(a, b):
    """The same answer, possibly at another precision: one contains the other as whole words."""
    a, b = plain(a), plain(b)
    return bool(a and b) and (f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def find_quote(quote, text):
    """Is `quote` in `text`? "exact": word for word, apart from case, punctuation and
    spacing. "close": nearly, with a few words dropped or changed. None: it is not there."""
    q, t = plain(quote).split(), plain(text).split()
    if not q:
        return None
    if f" {' '.join(q)} " in f" {' '.join(t)} ":
        return "exact"
    runs = SequenceMatcher(None, t, q, autojunk=False).get_matching_blocks()
    return "close" if sum(r.size for r in runs if r.size >= 3) >= CLOSE * len(q) else None


def forms(answers):
    """The distinct ways an answer is written, the most frequent first, leaving out any that
    a fuller one contains ("December 2009" beside "18 December 2009")."""
    written = {}
    for a in answers:
        written.setdefault(plain(a), a)
    counts = collections.Counter(plain(a) for a in answers)
    keys = [k for k in written if k and not any(k != other and f" {k} " in f" {other} " for other in written)]
    return [written[k] for k in sorted(keys, key=lambda k: -counts[k])] or [answers[0]]


MONTHS = {name.lower(): n for names in (calendar.month_name, calendar.month_abbr)
          for n, name in enumerate(names) if name}


def when(date):
    """A date as (year, month, day) for ordering, with 0 for a missing part; None without a year."""
    text = plain(date)
    iso = re.search(r"\b(1\d{3}|20\d{2}) (\d{1,2}) (\d{1,2})\b", text)  # 2024-03-05
    if iso:
        return tuple(map(int, iso.groups()))
    year = re.search(r"\b(1\d{3}|20\d{2})\b", text)
    if not year:
        return None
    month = next((MONTHS[word] for word in text.split() if word in MONTHS), 0)
    day = re.search(r"\b([0-3]?\d)(st|nd|rd|th)?\b", text.replace(year.group(), " "))
    return int(year.group()), month, int(day.group(1)) if day else 0


def json_values(text):
    """Every outermost JSON object or list in a model's reply, in order. A reply cut off by
    the token cap still gives the items that were complete."""
    values, at, decoder, opening = [], 0, json.JSONDecoder(), re.compile(r"[\[{]")
    while match := opening.search(text, at):
        try:
            value, at = decoder.raw_decode(text, match.start())
            values.append(value)
        except ValueError:
            at = match.start() + 1
    return values


def first(values, kind):
    """The first value of type `kind` (dict or list), or None."""
    return next((v for v in values if isinstance(v, kind)), None)


def number(value):
    """`value` as an int (the model may write 3 or "3"), or None."""
    return int(value) if str(value).strip().isdigit() else None


def components(nodes, edges):
    """The groups that `edges` join `nodes` into. Every node is in exactly one group."""
    root = {n: n for n in nodes}

    def find(n):
        while root[n] != n:
            n = root[n]
        return n

    for a, b in edges:
        root[find(a)] = find(b)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


# ---- The model ----

class Session:
    """One question's run: the model, what its calls cost, and the trace of every step.
    The trace's "calls" logs each call: its step, tokens, whether it hit its token cap, and
    whether the reply held any JSON."""

    def __init__(self, llm):
        self.llm = llm
        self.cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "capped": 0}
        self.trace = {"calls": []}

    def ask(self, prompt, max_tokens, **parts):
        """One call. Returns the JSON values in the reply, and the reply's text."""
        reply = self.llm.call([{"role": "user", "content": prompt.format(**parts)}],
                              max_tokens=max_tokens, **DECODING)
        values, capped = json_values(reply["text"]), reply["finish_reason"] == "length"
        self.cost["calls"] += 1
        self.cost["prompt_tokens"] += reply["prompt_tokens"]
        self.cost["completion_tokens"] += reply["completion_tokens"]
        self.cost["capped"] += capped
        self.trace["calls"].append({"step": STEPS[prompt], "prompt_tokens": reply["prompt_tokens"],
                                    "completion_tokens": reply["completion_tokens"], "capped": capped,
                                    "json": bool(values)})
        return values, reply["text"]


def show_docs(numbers, docs):
    return "\n".join(f"Document {d}: {docs[d - 1]['text']}" for d in numbers)


def show_answers(sides):
    """Answer clusters as the model is shown them. `sides` maps a letter to a cluster."""
    def show(letter, c):
        said = list(dict.fromkeys(card["attributed_to"] for card in c["cards"] if stated(card["attributed_to"])))
        return "\n".join([
            f"Answer {letter}: {c['answer']}",
            f"  about: {c['entity_name']}",
            f"  sense: {c['sense']}" + (" (the document does not state it)" if c["sense_unknown"] else ""),
            f"  as of: {c['date']}",
            f"  kind: {'opinion' if any(card['claim_kind'] == 'opinion' for card in c['cards']) else 'factual'}",
            *([f"  attributed to: {'; '.join(said)}"] if said else []),
            f"  independent sources: {c['independent']}",
            "  evidence: " + "; ".join(f'"{card["evidence"]}" [document {card["doc"]}]' for card in c["cards"][:3])])

    return "\n\n".join(show(letter, c) for letter, c in sides.items())


# ---- Steps 0 and 1: memory (optional) and intake ----

def recall(s, question):
    """Step 0, only with memory on. The model's own answers, recorded before it sees any
    document. Step 8 may use them as a last tiebreaker."""
    values, _ = s.ask(RECALL, 200, question=question)
    answers = (first(values, dict) or {}).get("answers")
    s.trace["memory"] = [
        {"about": str(a.get("about", "")), "answer": str(a["answer"]), "confidence": number(a.get("confidence")) or 0}
        for a in (answers if isinstance(answers, list) else []) if isinstance(a, dict) and stated(a.get("answer"))]
    return s.trace["memory"]


def intake(s, question):
    """Step 1. The question goes on unchanged. Two facts about it are recorded: the time it
    names, and whether its answer can change over time (assumed so if the reply can't be read)."""
    values, _ = s.ask(INTAKE, 60, question=question)
    data = first(values, dict) or {}
    time = data.get("time")
    s.trace["intake"] = {"time": str(time) if stated(time) else None,
                         "changes_over_time": data.get("changes_over_time") is not False}
    return s.trace["intake"]


# ---- Steps 3 and 4: fact cards, verified ----

def lenient_cards(text):
    """Cards from a reply that is not valid JSON, usually because of an unescaped quote
    inside a field: each field is read up to the next field's name or the closing brace."""
    names = "|".join(FIELDS)
    field = re.compile(rf'"({names})"\s*:\s*"(.*?)"\s*(?=,\s*"(?:{names})"\s*:|\}})', re.S)
    return [{f: v.replace('\\"', '"') for f, v in field.findall(card)}
            for card in re.split(r'(?="entity"\s*:)', text)[1:]]


def clean(raw):
    """A card with every field a one-line string; None if it has no answer or no quote."""
    if not isinstance(raw, dict):
        return None
    card = {f: " ".join(str("not stated" if raw.get(f) in (None, "") else raw[f]).split()) for f in FIELDS}
    card["claim_kind"] = "opinion" if "opinion" in card["claim_kind"].lower() else "factual"
    return card if stated(card["answer"]) and stated(card["evidence"]) else None


def extract(s, question, docs):
    """Steps 3 and 4. The model reads each document on its own and writes a fact card for
    each answer it gives. A card is kept only if its quote is really in the document.
    Returns the cards, and the numbers of the documents that are relevant."""
    cards, relevant, s.trace["extract"] = [], [], []
    for d, doc in enumerate(docs, 1):
        values, text = s.ask(EXTRACT, 1024, question=question, document=doc["text"])
        whole = next((v for v in values if isinstance(v, dict) and ("cards" in v or "relevant" in v)), None)
        if whole:
            found = whole.get("cards") if isinstance(whole.get("cards"), list) else []
        else:  # a bare list of cards, a reply that was cut off, or invalid JSON: take the cards that can be read
            loose = [x for v in values for x in (v if isinstance(v, list) else [v])]
            found = [x for x in loose if isinstance(x, dict) and "answer" in x] or lenient_cards(text)
        kept, dropped = [], []
        for card in filter(None, map(clean, found)):
            match = find_quote(card["evidence"], doc["text"])
            if match:
                kept.append({"id": len(cards) + len(kept) + 1, "doc": d, **card, "quote": match})
            else:
                dropped.append({**card, "why": "its quote is not in the document"})
        step = {"doc": d, "relevant": bool(kept) or not whole or whole.get("relevant") is not False,
                "cards": kept, "dropped": dropped}
        if not whole and not found:
            step["unread"] = text
        cards += kept
        relevant += [d] * step["relevant"]
        s.trace["extract"].append(step)
    return cards, relevant


# ---- Step 5: the evidence map ----

def independent(numbers, docs):
    """How many independent sources a set of documents amounts to. Copies and light edits of
    one text count once, and so do documents that name the same "source". An estimate:
    paraphrases and coordinated fakes still look independent."""
    def same(a, b):
        if a.get("source") and a.get("source") == b.get("source"):
            return True
        return SequenceMatcher(None, plain(a["text"]).split(), plain(b["text"]).split(), autojunk=False).ratio() >= COPY

    return len(components(numbers, [(a, b) for a, b in itertools.combinations(numbers, 2)
                                    if same(docs[a - 1], docs[b - 1])]))


def finish(c, docs):
    """Fill in what follows from a cluster's cards: its documents, answer, date and
    independent sources."""
    c["docs"] = sorted({card["doc"] for card in c["cards"]})
    c["forms"] = forms([card["answer"] for card in c["cards"]])
    c["answer"] = c["forms"][0]
    dates = collections.Counter(card["as_of"] for card in c["cards"] if stated(card["as_of"]))
    c["date"] = c.get("dated") or (dates.most_common(1)[0][0] if dates else "not stated")
    c["independent"] = independent(c["docs"], docs)
    return c


def merge(members, docs):
    """Several clusters as one: answers that agree, or the parts of one fuller answer."""
    if len(members) == 1:
        return members[0]
    return finish({**members[0], "cards": [card for m in members for card in m["cards"]],
                   "uncertain": sorted({u for m in members for u in m["uncertain"]})}, docs)


def regroup(clusters, docs):
    """Join clusters that share an entity and a sense and have compatible answers."""
    joined = True
    while joined:
        joined = False
        for a, b in itertools.combinations(clusters, 2):
            if (a["entity"] == b["entity"] and plain(a["sense"]) == plain(b["sense"])
                    and any(compatible(x, y) for x in a["forms"] for y in b["forms"])):
                a["cards"] += b["cards"]
                a["uncertain"] += b["uncertain"]
                a["sense_unknown"] = a["sense_unknown"] and b["sense_unknown"]
                finish(a, docs)
                clusters[:] = [c for c in clusters if c is not b]
                joined = True
                break


def group_by_text(cards):
    """The grouping by matching text alone, for when the model's grouping can't be read."""
    entities = {}
    for c in cards:
        senses = entities.setdefault(plain(c["entity"]), {"name": c["entity"], "senses": {}})["senses"]
        answers = senses.setdefault(plain(c["sense"]), {"name": c["sense"], "answers": {}})["answers"]
        answers.setdefault(plain(c["answer"]), []).append(c["id"])
    return [{"name": e["name"], "senses": [{"name": s["name"], "answers": list(s["answers"].values())}
                                           for s in e["senses"].values()]} for e in entities.values()]


def view(c):
    """A cluster for the trace: its cards by number."""
    return {"entity": c["entity_name"], "sense": c["sense"], "sense_unknown": c["sense_unknown"],
            "answer": c["answer"], "date": c["date"], "cards": [card["id"] for card in c["cards"]],
            "uncertain": c["uncertain"], "docs": c["docs"], "independent": c["independent"]}


def build_map(s, question, cards, docs):
    """Step 5. Group the cards by entity, then sense, then answer, and count the independent
    sources behind each answer. Returns the answer clusters."""
    grouping, uncertain, by = None, [], "model"
    if len(cards) > 1:
        shown = "\n".join(f"Card {c['id']} (from document {c['doc']})\n  entity: {c['entity']}\n  sense: {c['sense']}\n"
                          f"  answer: {c['answer']}\n  as of: {c['as_of']}\n  evidence: \"{c['evidence']}\"" for c in cards)
        values, _ = s.ask(MAP, 512, question=question, cards=shown)
        data = first(values, dict) or {}
        grouping = data.get("entities")
        uncertain = [number(u) for u in data["uncertain"]] if isinstance(data.get("uncertain"), list) else []
    if not isinstance(grouping, list):
        grouping, by = group_by_text(cards), "text match" if len(cards) > 1 else "nothing to group"

    left, clusters = {c["id"]: c for c in cards}, []

    def cluster(entity, name, sense, members):
        clusters.append({"entity": entity, "entity_name": name, "sense": sense if stated(sense) else "not stated",
                         "sense_unknown": False, "cards": members,
                         "uncertain": [c["id"] for c in members if c["id"] in uncertain]})

    for e, entity in enumerate(g for g in grouping if isinstance(g, dict)):
        senses = entity.get("senses") if isinstance(entity.get("senses"), list) else []
        for sense in (x for x in senses if isinstance(x, dict)):
            answers = sense.get("answers") if isinstance(sense.get("answers"), list) else []
            if answers and not any(isinstance(g, (list, dict)) for g in answers):
                answers = [answers]  # one flat list of card numbers is one group
            for group in answers:
                ids = group.get("cards") if isinstance(group, dict) else group
                members = [left.pop(i) for i in map(number, ids if isinstance(ids, list) else []) if i in left]
                if members:
                    cluster(e, str(entity.get("name") or members[0]["entity"]), str(sense.get("name")), members)
    for card in list(left.values()):  # a card the grouping left out stands alone
        cluster(f"card {card['id']}", card["entity"], card["sense"], [card])

    # A card with no stated sense joins its entity's largest sense, and is flagged.
    for e in dict.fromkeys(c["entity"] for c in clusters):
        size = collections.Counter()
        for c in clusters:
            if c["entity"] == e and stated(c["sense"]):
                size[c["sense"]] += len({card["doc"] for card in c["cards"]})
        for c in clusters:
            if c["entity"] == e and not stated(c["sense"]) and size:
                c["sense"], c["sense_unknown"] = size.most_common(1)[0][0], True

    for c in clusters:
        finish(c, docs)
    regroup(clusters, docs)
    s.trace["map"] = {"grouped_by": by, "clusters": [view(c) for c in clusters]}
    return clusters


# ---- Step 6: explain before accusing ----

def explain(s, question, clusters, docs, relevant):
    """Step 6. Where answers about the same entity and sense differ, look in the documents
    for a sentence that explains the difference. If one is really there, the answers are
    given the senses or dates it shows. The model may not make an explanation up."""
    s.trace["explain"] = []
    places = collections.defaultdict(list)
    for c in clusters:
        places[c["entity"], c["sense"]].append(c)
    for group in [g for g in places.values() if len(g) > 1]:
        sides = dict(zip(string.ascii_uppercase, group))
        values, _ = s.ask(EXPLAIN, 200, question=question, answers=show_answers(sides),
                          documents=show_docs(relevant, docs))
        data = first(values, dict) or {}
        labels = data.get("labels") if isinstance(data.get("labels"), dict) else {}
        labels = {k: str(v) for k, v in labels.items() if k in sides and stated(v)}
        d, quote = number(data.get("document")), str(data.get("quote") or "")
        found = (data.get("explained") is True and data.get("by") in ("sense", "time") and d in relevant
                 and len(set(map(plain, labels.values()))) > 1 and bool(find_quote(quote, docs[d - 1]["text"])))
        record = {"entity": group[0]["entity_name"], "sense": group[0]["sense"],
                  "answers": [c["answer"] for c in group], "explained": found}
        if found:
            record.update(by=data["by"], document=d, quote=quote, labels=labels)
            for letter, label in labels.items():
                if data["by"] == "sense":
                    sides[letter]["sense"], sides[letter]["sense_unknown"] = label, False
                else:
                    sides[letter]["dated"] = sides[letter]["date"] = label
            for c in group:
                c["explanation"] = record
        s.trace["explain"].append(record)
    regroup(clusters, docs)  # an explanation may have moved an answer next to a compatible one


# ---- Step 7: label relations ----

def classify(s, question, a, b):
    """The relation between two answer clusters, and the reason given. This is the part a
    trained classifier is to replace; for now the model is prompted. A reply that can't be
    read counts as a contradiction, the most serious relation."""
    found = a.get("explanation") if a.get("explanation") is b.get("explanation") else None
    explanation = f'"{found["quote"]}" [document {found["document"]}]' if found else "none"
    values, text = s.ask(LABEL, 120, question=question, answers=show_answers({"A": a, "B": b}),
                         explanation=explanation)
    data = first(values, dict) or {}
    loose = re.search(r'"relation"\s*:\s*"([^"]*)"', text)
    said = plain(data.get("relation") or (loose.group(1) if loose else ""))
    relation = next((r for r in RELATIONS if r == said), None) or next((r for r in RELATIONS if r in said), None)
    return relation or "contradiction", str(data.get("reason") or "")


def label(s, question, clusters):
    """Step 7. One of the seven relations for each pair of answer clusters. Pairs the map put
    under different entities are "different entity"; every other pair goes to the classifier."""
    relations, s.trace["label"] = {}, []
    for (i, a), (j, b) in itertools.combinations(enumerate(clusters), 2):
        if a["entity"] != b["entity"] and not a["uncertain"] and not b["uncertain"]:
            relations[i, j], record = "different entity", {"by": "map"}
        else:
            relations[i, j], reason = classify(s, question, a, b)
            record = {"by": "classifier", "reason": reason}
        s.trace["label"].append({"a": a["answer"], "b": b["answer"], "relation": relations[i, j], **record})
    return relations


# ---- Step 8: adjudicate contradictions ----

def adjudicate(s, question, sides, docs, relevant, facts, recalled, mode):
    """Step 8. Which of the contradicting answers in `sides` stands, and on what basis.
    Evidence is tried in a fixed order, and the first kind that leaves one answer decides.
    If nothing does: None in "disagree" mode; in "forced" mode the better-supported answer,
    with a basis that starts with "forced"."""
    letters = dict(zip(string.ascii_uppercase, sides))
    shown, live = show_answers(letters), list(sides)
    used = sorted({d for c in sides for d in c["docs"]})
    record = {"entity": sides[0]["entity_name"], "sense": sides[0]["sense"],
              "answers": [c["answer"] for c in sides], "tried": []}
    s.trace["adjudicate"].append(record)

    def settle(winner, basis):
        record.update(stands=winner["answer"] if winner else None, basis=basis)
        return winner, basis

    def pick(values, key):
        """The cluster a reply names by its letter ("A" or "Answer A"), or None."""
        named = re.fullmatch(r"(?:answer\s*)?([a-z])", str((first(values, dict) or {}).get(key)).strip(), re.I)
        return letters.get(named.group(1).upper()) if named else None

    # 1. Source information, when the documents carry any.
    if any(len(docs[d - 1]) > 1 for d in used):
        sources = "\n".join(f"Document {d}: " + "; ".join(f"{k}: {v}" for k, v in docs[d - 1].items() if k != "text")
                            for d in used)
        values, _ = s.ask(SOURCES, 120, question=question, answers=shown, sources=sources)
        favored = pick(values, "favors")
        record["tried"].append({"evidence": "source information", "favors": favored["answer"] if favored else None})
        if favored:
            return settle(favored, "source information")

    # 2. A fact in documents outside the disagreement that makes one answer impossible.
    #    One side's own documents can't rule out the other: that would be circular.
    outside = [d for d in relevant if d not in used]
    if outside:
        values, _ = s.ask(FACTS, 200, question=question, answers=shown, documents=show_docs(outside, docs))
        data = first(values, dict) or {}
        out, d, quote = pick(values, "impossible"), number(data.get("document")), str(data.get("quote") or "")
        ruled_out = bool(out) and d in outside and bool(find_quote(quote, docs[d - 1]["text"]))
        record["tried"].append({"evidence": "other facts", "rules_out": out["answer"] if ruled_out else None,
                                **({"document": d, "quote": quote} if ruled_out else {})})
        if ruled_out:
            live = [c for c in live if c is not out]
            if len(live) == 1:
                return settle(live[0], "other facts in the documents")

    # 3. A clear majority of independent sources. Cards whose entity is uncertain don't count.
    def certain(c):
        return independent(sorted({card["doc"] for card in c["cards"] if card["id"] not in c["uncertain"]}), docs)

    live.sort(key=certain, reverse=True)
    top, second = certain(live[0]), certain(live[1])
    record["tried"].append({"evidence": "majority", "independent_sources": {c["answer"]: certain(c) for c in live}})
    if MAJORITY and top > second and top >= CLEARLY_AHEAD * second:
        return settle(live[0], f"majority of independent sources ({top} against {second})")

    # 4. The model's memory: only if it is on, the model was confident, and the answer is fixed.
    if recalled is not None and not facts["changes_over_time"]:
        sure = [m["answer"] for m in recalled if m["confidence"] >= CONFIDENT]
        backed = [c for c in live if any(compatible(f, a) for f in c["forms"] for a in sure)]
        record["tried"].append({"evidence": "memory", "backs": [c["answer"] for c in backed]})
        if len(backed) == 1:
            return settle(backed[0], "the model's memory")

    if mode == "disagree":
        return settle(None, "sources disagree")

    # Forced: a choice has to be made. More independent sources, else the model's reading.
    if top > second:
        return settle(live[0], f"forced: more independent sources ({top} against {second})")
    knowledge = ("If the documents give no reason to prefer one answer, go by what you know."
                 if recalled is not None else "Judge by the documents alone.")
    values, text = s.ask(CHOOSE, 150, question=question, answers=shown, documents=show_docs(relevant, docs),
                         knowledge=knowledge)
    loose = re.search(r'"choice"\s*:\s*"?([A-Za-z])', text)
    chosen = pick(values, "choice") or (letters.get(loose.group(1).upper()) if loose else None)
    if any(chosen is c for c in live):
        return settle(chosen, "forced: the model's reading of the documents")
    return settle(live[0], "forced: nothing favors either answer, so the first one stands")


# ---- Step 9: the answer ----

def entry(c, basis=None, **more):
    """An answer cluster as an entry of the answer."""
    basis = basis or c.get("basis") or (
        f"{c['independent']} independent sources agree" if c["independent"] > 1 else "one source")
    e = {"entity": c["entity_name"], "sense": c["sense"], "answers": c["forms"], "date": c["date"],
         "sources": c["docs"], "independent_sources": c["independent"], "basis": basis}
    if c.get("outweighed"):
        e["outweighed"] = c["outweighed"]
    return {**e, **more}


def changed(members, facts, docs):
    """Values that changed over time: the latest (or the one at the time the question names),
    with the others and their dates."""
    dated = sorted((m for m in members if when(m["date"])), key=lambda m: when(m["date"]), reverse=True)
    if len(dated) < len(members):
        return entry(merge(members, docs), "the value changed over time, but not every value is dated, "
                                           "so the latest can't be told")
    asked = when(facts["time"] or "")
    at = next((m for m in dated if when(m["date"])[0] <= asked[0]), None) if asked else None
    main = at or dated[0]
    return entry(main, "the value at the time the question names" if at else "the latest dated value",
                 other_times=[{"answers": m["forms"], "date": m["date"], "sources": m["docs"]}
                              for m in dated if m is not main])


def resolve(s, question, clusters, relations, docs, relevant, facts, recalled, mode):
    """Steps 8 and 9. Turn the labelled answer clusters into the entries of the answer."""
    s.trace["adjudicate"] = []
    # Clusters that agree become one group. Two groups relate by their most serious pair.
    groups, home = [], {}
    for members in components(range(len(clusters)), [p for p, r in relations.items() if r == "agree"]):
        for i in members:
            home[i] = len(groups)
        groups.append(merge([clusters[i] for i in members], docs))
    between = {}
    for (i, j), r in relations.items():
        pair = tuple(sorted((home[i], home[j])))
        if pair[0] != pair[1]:
            between[pair] = max(between.get(pair, r), r, key=RELATIONS.index)

    def linked(relation, alive):
        """Sets of two or more groups in `alive` joined by `relation`."""
        edges = [p for p, r in between.items() if r == relation and p[0] in alive and p[1] in alive]
        return [g for g in components(alive, edges) if len(g) > 1]

    entries, alive = [], list(range(len(groups)))
    for members in linked("contradiction", alive):
        sides = [groups[g] for g in members]
        winner, basis = adjudicate(s, question, sides, docs, relevant, facts, recalled, mode)
        alive = [g for g in alive if g not in members or groups[g] is winner]
        if winner:
            winner["basis"] = basis
            winner["outweighed"] = [{"answers": c["forms"], "sources": c["docs"]} for c in sides if c is not winner]
        else:
            entries.append({"entity": sides[0]["entity_name"], "sense": sides[0]["sense"], "answers": [],
                            "sources": sorted({d for c in sides for d in c["docs"]}), "basis": basis,
                            "sides": [{"answers": c["forms"], "sources": c["docs"],
                                       "independent_sources": c["independent"]} for c in sides]})
    for relation in ("different time", "complementary", "opinion"):
        for members in linked(relation, alive):
            parts = [groups[g] for g in members]
            alive = [g for g in alive if g not in members]
            if relation == "different time":
                entries.append(changed(parts, facts, docs))
            elif relation == "complementary":
                entries.append(entry(merge(parts, docs), "complementary answers, merged"))
            else:
                entries.append(entry(merge(parts, docs), "opinions: each position is given", positions=[
                    {"answers": c["forms"], "claim": c["cards"][0]["claim"], "sources": c["docs"]} for c in parts]))
    entries += [entry(groups[g]) for g in alive]
    return sorted(entries, key=lambda e: e["sources"])


def respond(entries):
    """The answer in words, built from the entries: one line each, with its sources."""
    def cite(sources):
        return "[" + ", ".join(map(str, sources)) + "]"

    lines = []
    for e in entries:
        about = e["entity"] + (f", {e['sense']}" if stated(e["sense"]) else "")
        if "sides" in e:
            lines.append(f"{about}: sources disagree: " + " or ".join(
                f"{' / '.join(side['answers'])} {cite(side['sources'])}" for side in e["sides"])
                + ". Nothing in the documents settles it.")
        elif "positions" in e:
            lines.append(f"{about}: opinions differ. " + " ".join(
                f"{p['claim']} {cite(p['sources'])}" for p in e["positions"]))
        else:  # " / " between ways of writing one answer, " and " between the parts of a merged one
            line = f"{about}: {(' and ' if e['basis'].startswith('complementary') else ' / ').join(e['answers'])}"
            line += f" (as of {e['date']})" if stated(e["date"]) else ""
            line += f" {cite(e['sources'])}. Basis: {e['basis']}."
            for other in e.get("other_times", []):
                line += f" At another time ({other['date']}): {' / '.join(other['answers'])} {cite(other['sources'])}."
            for other in e.get("outweighed", []):
                line += f" Outweighed: {' / '.join(other['answers'])} {cite(other['sources'])}."
            lines.append(line)
    return "\n".join(lines) or "Cannot determine: no document gives an answer."


def run(llm, question, documents, mode="disagree", memory=False):
    """Answer `question` from `documents`: dicts with "text", plus whatever is known about
    the source ("source", "date", "kind", ...). Returns the entries, a status (answered,
    partly answered, contested or cannot determine), the response in words, a question to
    ask back when there are several readings, the cost of the calls, and the trace: the
    recorded output of every step."""
    assert mode in MODES, mode
    s = Session(llm)
    recalled = recall(s, question) if memory else None  # before any document is read
    facts = intake(s, question)
    cards, relevant = extract(s, question, documents)
    clusters = build_map(s, question, cards, documents)
    explain(s, question, clusters, documents, relevant)
    relations = label(s, question, clusters)
    entries = resolve(s, question, clusters, relations, documents, relevant, facts, recalled, mode)

    contested = sum("sides" in e for e in entries)
    status = ("cannot determine" if not entries else "answered" if not contested
              else "contested" if contested == len(entries) else "partly answered")
    readings = list(dict.fromkeys(e["entity"] + (f" ({e['sense']})" if stated(e["sense"]) else "")
                                  for e in entries if "sides" not in e))
    ask_back = f"Did you mean {', '.join(readings[:-1])} or {readings[-1]}?" if len(readings) > 1 else None
    return {"entries": entries, "status": status, "response": respond(entries), "ask_back": ask_back,
            **s.cost, "trace": s.trace}


def main():
    from src.interface import LLM, MODELS  # here, so that importing the pipeline stays light

    parser = argparse.ArgumentParser(description="Answer one question from its documents with the pipeline.")
    parser.add_argument("file", nargs="?", help='a JSON file {"question": "...", "documents": [{"text": "..."}]}; '
                                                "without it, the Avatar example from the plan")
    parser.add_argument("--mode", choices=MODES, default="disagree",
                        help='for a contradiction nothing settles: "disagree" reports both sides (default), '
                             '"forced" picks the better-supported one')
    parser.add_argument("--memory", action="store_true",
                        help="let the model's own knowledge break ties, as a last resort")
    parser.add_argument("--trace", action="store_true", help="also print what every step recorded")
    parser.add_argument("--backend", choices=list(MODELS), default="vllm")
    args = parser.parse_args()
    task = EXAMPLE
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            task = json.load(f)
    with LLM(args.backend) as llm:
        result = run(llm, task["question"], task["documents"], args.mode, args.memory)
    if args.trace:
        print(json.dumps(result["trace"], indent=2, ensure_ascii=False))
    print(f"\nQuestion: {task['question']}\n{result['response']}\nStatus: {result['status']}")
    if result["ask_back"]:
        print(f"Or ask back: {result['ask_back']}")
    print(f"Cost: {result['calls']} calls, {result['prompt_tokens']} input and "
          f"{result['completion_tokens']} output tokens")


if __name__ == "__main__":
    main()
