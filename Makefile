.PHONY: test
test:
	smartsensor \
		--train-image-data . \
		--feature . \
		--norm none \
		--norm delta \
		--degree 2 \
		--out .