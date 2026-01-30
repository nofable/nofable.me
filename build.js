const { Marked } = require("marked");
const { markedHighlight } = require("marked-highlight");
const fs = require("fs");
const Handlebars = require("handlebars");
const hljs = require('highlight.js');

// create paths
const sitemapPath = `${__dirname}/sitemap.json`;
const postsDir = `${__dirname}/posts`;
const postTemplatePath = `${__dirname}/templates/post.html`;
const homeTemplatePath = `${__dirname}/templates/home.html`;
const outputDir = `${__dirname}/out`;

const sitemap = JSON.parse(fs.readFileSync(sitemapPath).toString());

// Mke the out directory if it doesn't exist
if (fs.existsSync(outputDir)) {
  fs.rmSync(outputDir, { recursive: true, force: true });
}
fs.mkdirSync(outputDir);

// go through posts, create a html page for each one
for (index in sitemap.posts) {
  const post = sitemap.posts[index];
  const contents = fs.readFileSync(`${postsDir}/${post.filename}`).toString();

  // run the markdown to html conversion
  const marked = new Marked(
    // the following does code block syntax highlighting
    markedHighlight({
      emptyLangClass: 'hljs',
      langPrefix: 'hljs language-',
      highlight(code, lang, info) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
      }
    }),

  );

  // leaving this as example renderer for when i do progressive images
  const renderer = {
    // image({ href }) {
    //   return `<img src="${href}" />`;
    // }
  }
  marked.use({ renderer })
  const htmlPost = marked.parse(contents);

  // get the template for post pages
  const source = fs.readFileSync(postTemplatePath).toString();
  // compile the handlebars template
  const template = Handlebars.compile(source);
  // create the final html page
  const result = template({ ...post, htmlPost });
  // write to the output directory
  fs.writeFileSync(`${outputDir}/${post.slug}.html`, result);
}

// Generate home page
sitemap.home.slug
const source = fs.readFileSync(homeTemplatePath).toString();
const template = Handlebars.compile(source);
// pass in all the posts
const result = template({ posts: sitemap.posts });
// write to the output diretory
fs.writeFileSync(`${outputDir}/${sitemap.home.slug}`, result);

// copy css files across to the output directory
fs.cpSync("./css", `${outputDir}`, { recursive: true });
// copy media files across to the output directory
fs.cpSync("./media", `${outputDir}/media`, { recursive: true });
