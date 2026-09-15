(() => {
  const select = document.getElementById("paper-sort");
  const list = document.getElementById("paper-list");

  if (!select || !list) return;

  const papers = Array.from(list.querySelectorAll(".paper-item"));

  const render = () => {
    const direction = select.value === "title-asc" ? 1 : -1;
    const sorted = [...papers].sort(
      (a, b) => direction * a.dataset.title.localeCompare(b.dataset.title, "en")
    );
    sorted.forEach((paper) => list.appendChild(paper));
  };

  select.addEventListener("change", render);
  render();
})();
