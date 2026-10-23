"""Cost estimation for the provider agents (see ``agents/provider/cost_estimators.py``).

``prompt``       LLM-Estimate's prompt, response schema, parser and cost model
``regression``   Regression's fit on countable request features
``requirement``  what an estimator sees of a request
``llm_client``   OpenRouter transport, repair rules and the call cache
"""
