.PHONY: doc tutorial travis_tests

ci_tests: bids/tests/data/bids-examples
	pytest --doctest-modules -n 2 -v --cov bids --cov-config .coveragerc --cov-report xml:cov.xml bids

tutorial:
	jupyter nbconvert --execute examples/pybids_tutorial.ipynb --to html

doc:
	$(MAKE) -C doc html

bids/tests/data/bids-examples:
	git clone https://github.com/bids-standard/bids-examples.git --depth 1 bids/tests/data/bids-examples