(() => {
  if (!document.querySelector(".equation-tex, .math-inline")) return;

  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true
    },
    chtml: {
      displayAlign: "center",
      displayIndent: "0"
    },
    options: {
      enableMenu: false
    }
  };

  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js";
  script.async = true;
  script.id = "MathJax-script";
  script.addEventListener("load", () => {
    document.documentElement.classList.add("mathjax-ready");
  });
  script.addEventListener("error", () => {
    document.documentElement.classList.add("mathjax-failed");
  });
  document.head.appendChild(script);
})();
