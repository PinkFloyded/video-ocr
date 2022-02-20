lint:
	flake8 --ignore E501

formattingcheck:
	black . --check

check: lint formattingcheck
	

