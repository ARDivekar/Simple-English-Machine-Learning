---
description: Adding jupyter notebooks
---

# Jupyter notebooks

Jupyter notebooks are an awesome way to show code and graphs. Combined with Bokeh, which renders HTML in the browser, you end up with some very nice data science notebooks.

You can export jupyter notebooks using a command like this one:

```sh
jupyter nbconvert --to html Bokeh.ipynb --output Bokeh_notebook.html
```

E.g. [Bokeh_notebook.html](./Bokeh_notebook.html)

Notebooks can also be included inline, using the ["include-html" plugin](https://github.com/Bandwidth/gitbook-plugin-include-html).

Example (note: this only seems to work when you build locally, and not on gitbook.com):

!INCLUDE "./Bokeh_notebook.html"

Side note: you can import file contents directly into your markdown during processing, as described [here](https://toolchain.gitbook.com/templating/conrefs.html).