"""Expose the V1 backend's implemented whitespace option through its older CLI."""


def main():
    import uvloop
    from vllm.entrypoints.openai.api_server import (
        cli_env_setup,
        make_arg_parser,
        validate_parsed_serve_args,
        run_server,
    )
    from vllm.utils import FlexibleArgumentParser

    cli_env_setup()
    parser = make_arg_parser(
        FlexibleArgumentParser(description="Project auxiliary compact-JSON server")
    )
    # Installed backend_xgrammar.py supports this option, but the older CLI
    # choices list omits suffix options. Only this local parser instance changes.
    for action in parser._actions:
        if action.dest == "guided_decoding_backend":
            action.choices = list(action.choices or []) + [
                "xgrammar:disable-any-whitespace"
            ]
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    if args.guided_decoding_backend != "xgrammar:disable-any-whitespace":
        raise ValueError("Auxiliary service requires xgrammar:disable-any-whitespace")
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
