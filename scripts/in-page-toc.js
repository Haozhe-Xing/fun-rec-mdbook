/**
 * In-Page Table of Contents — 右侧章节内导航
 *
 * 自动提取当前页面的 h2 / h3 标题，生成粘性侧边栏。
 * - 滚动时高亮当前可见章节
 * - 点击跳转
 * - 少于 3 个标题时自动隐藏（首页、Part 概览等短页）
 * - 移动端自动隐藏
 */
(function () {
  "use strict";

  var MIN_HEADINGS = 3;
  var SELECTOR = ".content h2, .content h3";
  var ACTIVE_CLASS = "iptoc-active";

  // ── 等待 mdBook 页面渲染完成 ──
  function waitForContent(cb) {
    var el = document.querySelector(".content");
    if (el && el.children.length > 0) {
      cb(el);
      return;
    }
    setTimeout(function () { waitForContent(cb); }, 100);
  }

  function buildToc(contentEl) {
    var headings = contentEl.querySelectorAll(SELECTOR);

    // 标题太少就不显示 TOC
    if (headings.length < MIN_HEADINGS) return;

    // ── 创建容器结构 ──
    // 将 .content 包裹进 .iptoc-wrapper，旁边放 .iptoc-sidebar
    var wrapper = document.createElement("div");
    wrapper.className = "iptoc-wrapper";
    contentEl.parentNode.insertBefore(wrapper, contentEl);
    wrapper.appendChild(contentEl);

    var sidebar = document.createElement("nav");
    sidebar.className = "iptoc-sidebar";
    sidebar.setAttribute("aria-label", "章节内目录");
    wrapper.appendChild(sidebar);

    // ── 标题栏 ──
    var header = document.createElement("div");
    header.className = "iptoc-header";
    header.textContent = "本节目录";
    sidebar.appendChild(header);

    // ── 列表 ──
    var ul = document.createElement("ul");
    ul.className = "iptoc-list";
    sidebar.appendChild(ul);

    var idMap = {}; // 去重：防止重复 id

    headings.forEach(function (h, idx) {
      // 确保每个 heading 有 id（mdBook 会给，但保险起见）
      if (!h.id) {
        var raw = h.textContent.trim().replace(/\s+/g, "-");
        h.id = raw.toLowerCase().replace(/[^\w\u4e00-\u9fff\-]/g, "");
      }
      if (!h.id) h.id = "heading-" + idx;

      // 去重后缀
      if (idMap[h.id] !== undefined) {
        idMap[h.id]++;
        h.id = h.id + "-" + idMap[h.id];
      } else {
        idMap[h.id] = 0;
      }

      var li = document.createElement("li");
      li.className = h.tagName.toLowerCase() === "h3" ? "iptoc-h3" : "iptoc-h2";

      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent.trim();
      a.dataset.targetId = h.id;

      a.addEventListener("click", function (e) {
        e.preventDefault();
        var target = document.getElementById(this.dataset.targetId);
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });

      li.appendChild(a);
      ul.appendChild(li);
    });

    // ── 滚动高亮 ──
    var tocLinks = ul.querySelectorAll("a");
    var headingEls = Array.prototype.slice.call(headings);

    function updateActive() {
      var scrollY = window.scrollY;
      var offset = 120; // 提前触发高亮

      var current = null;
      for (var i = headingEls.length - 1; i >= 0; i--) {
        if (headingEls[i].getBoundingClientRect().top + scrollY <= scrollY + offset) {
          current = headingEls[i].id;
          break;
        }
      }

      tocLinks.forEach(function (link) {
        if (current && link.dataset.targetId === current) {
          link.classList.add(ACTIVE_CLASS);
        } else {
          link.classList.remove(ACTIVE_CLASS);
        }
      });
    }

    window.addEventListener("scroll", updateActive, { passive: true });
    updateActive(); // 初始化一次
  }

  // ── 启动 ──
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      waitForContent(buildToc);
    });
  } else {
    waitForContent(buildToc);
  }
})();
