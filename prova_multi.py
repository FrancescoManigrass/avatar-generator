import concurrent.futures
from time import sleep, time


def some_function(_a, _b, _c):
    sleep(1)
    return (_a + _b) * _c


def some_function_wrapper(args):
    return some_function(*args)


def main():
    st = time()
    print(f"Starting...")
    with concurrent.futures.ProcessPoolExecutor() as _executor:
        pool = [
            _executor.submit(some_function_wrapper, args=(a, b, c))
            for a in range(3) for b in range(3) for c in range(3)
        ]
        k = 0
        for result in concurrent.futures.as_completed(pool):
            print(f"{k}\t{result.result()}")
            k += 1
    print(f"Fin t={time() - st}")


if __name__ == "__main__":
    main()