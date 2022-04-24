.PHONY: doc tutorial travis_tests

ci_tests:
	pytest --doctest-modules -n 2 -v --cov bids --cov-config .coveragerc --cov-report xml:cov.xml bids

tutorial:
	jupyter nbconvert --execute examples/pybids_tutorial.ipynb --to html

doc:
	$(MAKE) -C doc html
