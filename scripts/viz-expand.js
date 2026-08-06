// 交互式可视化 Lightbox：点击章节中的 iframe 可放大查看
(function () {
  function buildLightbox() {
    if (document.getElementById("viz-lightbox")) return;
    var lb = document.createElement("div");
    lb.id = "viz-lightbox";
    lb.style.cssText =
      "position:fixed;inset:0;background:rgba(15,23,42,0.85);" +
      "display:none;align-items:center;justify-content:center;z-index:9999;padding:4vh 4vw;";
    var frame = document.createElement("iframe");
    frame.style.cssText = "width:100%;height:100%;border:none;border-radius:12px;background:#0f172a;";
    var close = document.createElement("button");
    close.textContent = "✕ 关闭";
    close.style.cssText =
      "position:absolute;top:3vh;right:3vw;padding:8px 16px;border:none;border-radius:20px;" +
      "background:#4A6CF7;color:#fff;font-size:0.9em;cursor:pointer;font-weight:600;";
    lb.appendChild(frame);
    lb.appendChild(close);
    document.body.appendChild(lb);
    close.onclick = function () { lb.style.display = "none"; frame.src = ""; };
    lb.onclick = function (e) { if (e.target === lb) { lb.style.display = "none"; frame.src = ""; } };
  }

  function bind() {
    buildLightbox();
    var lb = document.getElementById("viz-lightbox");
    var frame = lb.querySelector("iframe");
    document.querySelectorAll("iframe[src*='viz/']").forEach(function (ifr) {
      if (ifr.dataset.expanded) return;
      ifr.dataset.expanded = "1";
      ifr.style.cursor = "zoom-in";
      ifr.addEventListener("click", function () {
        frame.src = ifr.src;
        lb.style.display = "flex";
      });
    });
  }

  if (document.readyState !== "loading") bind();
  else document.addEventListener("DOMContentLoaded", bind);
})();
