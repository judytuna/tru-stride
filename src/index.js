export default {
  async fetch(request) {
    const html = `<!DOCTYPE html>
    <head>
      <script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/5.35.0/gradio.js"></script>
    </head>
    <body>
    <gradio-app src="https://intuitivehorseware-tru-stride-analyzer.hf.space"></gradio-app>
    </body>`;

    return new Response(html, {
      headers: {
        "content-type": "text/html;charset=UTF-8",
      },
    });
  },
};
