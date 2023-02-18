import warnings

warnings.simplefilter("ignore", UserWarning)


def entry_point():
    from tldream.server import main

    main()
