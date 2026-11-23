"""Selected, dataset-separated score tables; rebuilds the Drive PDF."""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/Current_Results_20261117/Current_Results_Tables.pdf"
FULL = ROOT / "analysis/model_results_20261116/All_Full_Benchmark_Results.csv"
pdfmetrics.registerFont(TTFont("DV", str(ROOT.parent / "assets/fonts/DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-Bold", str(ROOT.parent / "assets/fonts/DejaVuSans-Bold.ttf")))
W,H=landscape(A3)
M=32


def read(p):
    with p.open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))


def line(c,x,y,s,size=8,bold=False,color="#263746"):
    c.setFillColor(colors.HexColor(color));c.setFont("DV-Bold" if bold else "DV",size);c.drawString(x,y,s)


def page(c,title,subtitle,num):
    c.setFillColor(colors.HexColor("#17354a"));c.rect(0,H-73,W,73,fill=1,stroke=0)
    line(c,M,H-34,title,16,True,"#ffffff")
    line(c,M,H-53,subtitle,8,False,"#ffffff")
    c.setStrokeColor(colors.HexColor("#b8c6cf"));c.line(M,30,W-M,30)
    line(c,M,18,"Source: saved project score CSVs. Observed exploratory results.",7)
    c.setFont("DV",7);c.drawRightString(W-M,18,str(num))


WIDTHS=[380,70,160,160,160,90]
HEADERS=["Method / model","N","Answer match","Strict set","Wrong accepted","Failed"]


def table(c,y,title,rows):
    line(c,M,y,title,10.5,True)
    best={j:(max if j in (2,3) else min)(int(r[j].split()[0]) for r in rows) for j in (2,3,4)}
    y-=13; rh=20
    c.setFillColor(colors.HexColor("#dfe8ee"));c.rect(M,y-rh+3,sum(WIDTHS),rh,fill=1,stroke=0)
    x=M
    for h,w in zip(HEADERS,WIDTHS):line(c,x+4,y-rh+7,h,7.4,True);x+=w
    y-=rh
    for i,r in enumerate(rows):
        if i%2:
            c.setFillColor(colors.HexColor("#f1f5f7"));c.rect(M,y-rh+3,sum(WIDTHS),rh,fill=1,stroke=0)
        x=M
        for j,(v,w) in enumerate(zip(r,WIDTHS)):
            highlight=j in best and int(v.split()[0])==best[j]
            line(c,x+4,y-rh+7,str(v),7.7,highlight,"#096a4e" if highlight else "#263746");x+=w
        y-=rh
    return y-16


def full_row(r):
    name={"Qwen2.5-7B-Instruct":"Qwen2.5-7B","Qwen3-32B":"Qwen3-32B","Qwen3.5-35B-A3B":"Qwen3.5-35B-A3B"}[r["model"]]
    method=r["method"]
    return [name+" | "+method,r["n"],f'{r["answer_match"]} ({float(r["answer_match_pct"]):.1f}%)',
            f'{r["strict_set"]} ({float(r["strict_set_pct"]):.1f}%)',
            f'{r["labeled_wrong"]} ({float(r["labeled_wrong_pct"]):.1f}%)',r["failed"]]


def main():
    full=read(FULL)
    assert len(full)==75
    c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
    c.setTitle("Full-Dataset Results by Dataset and Model")
    full_baselines=["Astute RAG (structured)","FaithfulRAG (adapted)","MADAM-RAG","Plain RAG",
                    "RobustRAG (keyword)","Self-reflection","Teammate Evidence Map",
                    "Teammate Evidence Map + Memory","Teammate Plain RAG (128 tokens)",
                    "TrustRAG (adapted)","TruthfulRAG (label-free)"]
    seven=full_baselines+["Docs + Cards (Ours)","Memory + Cards (Ours)","Evidence Pruning (Ours)"]
    other=["Plain RAG","Docs + Cards (Ours)","Memory + Cards (Ours)","Evidence Pruning (Ours)"]
    models=["Qwen2.5-7B-Instruct","Qwen3-32B","Qwen3.5-35B-A3B"]
    for page_no,ds,n in [(1,"RAMDocs",500),(2,"RGB",100),(3,"ConflictBank-500",500)]:
        page(c,f"{ds} | full dataset",f"{n} questions per method · three tested model backbones · full-benchmark experiment",page_no)
        y=H-91
        for model in models:
            d={r["method"]:r for r in full if r["dataset"]==ds and r["model"]==model}
            methods=seven if model==models[0] else other
            assert all(int(d[m]["n"])==n for m in methods)
            y=table(c,y,model+" | "+str(n)+" questions",[full_row(d[m]) for m in methods])
        line(c,M,y-1,"Bold green = best within each model table (ties included). Full-dataset runs used differing generation budgets across methods.",7.5)
        c.showPage()
    c.save();print(OUT)


if __name__=="__main__":main()
