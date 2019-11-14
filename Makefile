.PHONY: doc tutorial travis_tests

travis_tests:
	pytest -n 2 -v --cov bids --cov-config .coveragerc --cov-report xml:cov.xml bids

tutorial:
	jupyter nbconvert --execute examples/pybids_tutorial.ipynb

doc:
	$(MAKE) -C doc html
