---
description: Adding jupyter notebooks
---

# Jupyter

Jupyter notebooks are an awesome way to show code and graphs. Combined with Bokeh, which renders HTML in the browser, you end up with some very nice data science notebooks.

You can export jupyter notebooks using a command like this one:

```bash
jupyter nbconvert --to html Bokeh.ipynb --output Bokeh_notebook.html
```

E.g. [Bokeh\_notebook.html](https://github.com/ARDivekar/Simple-English-Machine-Learning/tree/91c8a3a232737008eed25b285974ac641bed279b/meta/Bokeh_notebook.html)

Notebooks can also be included inline, using the ["include-html" plugin](https://github.com/Bandwidth/gitbook-plugin-include-html).

Example \(note: this only seems to work when you build locally, and not on gitbook.com\):

!INCLUDE "./Bokeh\_notebook.html"

Side note: you can import file contents directly into your markdown during processing, as described [here](https://toolchain.gitbook.com/templating/conrefs.html).

