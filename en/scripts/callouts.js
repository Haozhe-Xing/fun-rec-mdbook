/**
 * Semantic Callouts — 按引用块首字符自动分类并上色
 *
 * 全书 58 个文件用 6+ 种语义标记（📝前置 / 💡提示 / ⚠️警告 /
 * 🧠心智模型 / 📊数据 / 🤔疑问 / **Analysis:**），但 mdBook 默认把它们
 * 全部渲染成同一种蓝边 blockquote。本脚本扫描 .content 内的引用块，
 * 按首字符 / 首 <strong> 打上 .callout-* 类，由 website.css 着色。
 *
 * 纯前端增强，不改动任何 markdown 源文件。
 */
(function () {
  "use strict";

  var RULES = [
    { test: /^📝/, cls: "callout-info" },
    { test: /^💡/, cls: "callout-tip" },
    { test: /^⚠️/, cls: "callout-warn" },
    { test: /^🧠/, cls: "callout-mental" },
    { test: /^📊/, cls: "callout-stat" },
    { test: /^🤔/, cls: "callout-question" }
  ];

  function init() {
    var quotes = document.querySelectorAll(".content blockquote");
    Array.prototype.forEach.call(quotes, function (q) {
      if (q.dataset.callout) return;
      q.dataset.callout = "1";

      var txt = (q.textContent || "").trim();

      // **Analysis:** 在渲染后成为 <strong>Analysis:</strong>
      var strong = q.querySelector("strong");
      if (strong && /^Analysis:/.test(strong.textContent.trim())) {
        q.classList.add("callout-analysis");
        return;
      }

      for (var i = 0; i < RULES.length; i++) {
        if (RULES[i].test.test(txt)) {
          q.classList.add(RULES[i].cls);
          return;
        }
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
