import os

from dotenv import load_dotenv
import robin_stocks.robinhood as rh

from scheduler import start_scheduler

load_dotenv()


def main() -> None:
    rh.login(
        username=os.environ["ROBINHOOD_USERNAME"],
        password=os.environ["ROBINHOOD_PASSWORD"],
        store_session=True,
    )
    print("Robinhood authenticated.")

    watchlist = [t.strip() for t in os.environ.get("WATCHLIST", "AAPL,MSFT,NVDA").split(",")]
    start_scheduler(watchlist)


if __name__ == "__main__":
    main()
