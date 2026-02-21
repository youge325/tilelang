# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html-en html-zh html-all

# The "clean" target is updated to remove the autoapi generated files as well.
# Run "make clean" to ensure a completely fresh build.
clean:
	rm -rf $(BUILDDIR) autoapi

# Build English documentation
html-en:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/en" $(SPHINXOPTS) $(O)

# Build Chinese documentation
html-zh:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)/zh_CN" "$(BUILDDIR)/zh_CN" $(SPHINXOPTS) $(O)

# Build both English and Chinese, then generate root redirect page
html-all: html-en html-zh
	@echo '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0;url=en/html/index.html"><script>window.location.href="en/html/index.html";</script></head><body><p><a href="en/html/index.html">English</a> | <a href="zh_CN/html/index.html">中文</a></p></body></html>' > $(BUILDDIR)/index.html
	@echo "Build complete for EN and ZH_CN."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
