"""Entry-point that forces WindowsSelectorEventLoopPolicy before uvicorn starts.

psycopg3 async cannot use the ProactorEventLoop that uvicorn sets by default
on Windows. Setting the policy here — before uvicorn.run() is called — ensures
the SelectorEventLoop is in place for the lifetime of the process.
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402 — import after policy is set

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
