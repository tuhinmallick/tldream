import warnings

warnings.simplefilter("ignore", UserWarning)

from tldream.server import main


def entry_point():
    main()
