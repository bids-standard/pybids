.PHONY: doc travis_tests

ci_tests:
	pytest --doctest-modules -n 2 -v --cov bids --cov-config .coveragerc --cov-report xml:cov.xml bids

doc:
	$(MAKE) -C doc html
