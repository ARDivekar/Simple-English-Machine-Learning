---
description: This page is about this Gitbook and how to set it up.
---

# Gitbook

Gitbook is a pretty nice tool, which allows you write docs in markdown, HTML, LaTeX, etc. and host them on gitbook.com

You can also just host your code on Github, and get people to pull the repo, then build the markdown to HTML locally on their computers.

For this repo, you just have to run `npm install --save-dev` and `npm run docs:watch`, and things should work.

The tutorial I followed is [this one](https://medium.com/@gpbl/how-to-use-gitbook-to-publish-docs-for-your-open-source-npm-packages-465dd8d5bfba), which sets up a few nice npm scripts like `docs:watch`. 
- Note, to get the `npm run docs:watch` command to work, I kept getting an `ENOSPC` Node.js error, which I solved using [this](https://github.com/facebook/jest/issues/3254#issuecomment-297869853).
