// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="index.html">🏠 首页</a></li><li class="chapter-item expanded affix "><li class="part-title">上篇 · 判别式推荐主线</li><li class="chapter-item expanded "><a href="part1-introduction/index.html"><strong aria-hidden="true">1.</strong> 推荐系统全景</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part1-introduction/recommender-system-basics.html"><strong aria-hidden="true">1.1.</strong> 推荐系统是什么</a></li><li class="chapter-item expanded "><a href="part1-introduction/book-overview.html"><strong aria-hidden="true">1.2.</strong> 本书概览与技术地图</a></li><li class="chapter-item expanded "><a href="part1-introduction/feature-embedding-basics.html"><strong aria-hidden="true">1.3.</strong> 特征与 Embedding 入门</a></li></ol></li><li class="chapter-item expanded "><a href="part2-retrieval/index.html"><strong aria-hidden="true">2.</strong> 快速候选召回</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part2-retrieval/collaborative-filtering.html"><strong aria-hidden="true">2.1.</strong> 协同过滤</a></li><li class="chapter-item expanded "><a href="part2-retrieval/vector-recall-i2i.html"><strong aria-hidden="true">2.2.</strong> 向量召回 (I2I)</a></li><li class="chapter-item expanded "><a href="part2-retrieval/two-tower.html"><strong aria-hidden="true">2.3.</strong> 双塔模型 (U2I)</a></li><li class="chapter-item expanded "><a href="part2-retrieval/sequence-recall.html"><strong aria-hidden="true">2.4.</strong> 序列召回</a></li><li class="chapter-item expanded "><a href="part2-retrieval/streaming-index.html"><strong aria-hidden="true">2.5.</strong> 流式索引召回</a></li></ol></li><li class="chapter-item expanded "><a href="part3-ranking/index.html"><strong aria-hidden="true">3.</strong> 精准偏好预测</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part3-ranking/wide-and-deep.html"><strong aria-hidden="true">3.1.</strong> Wide &amp; Deep</a></li><li class="chapter-item expanded "><a href="part3-ranking/feature-crossing.html"><strong aria-hidden="true">3.2.</strong> 特征交叉</a></li><li class="chapter-item expanded "><a href="part3-ranking/sequence-modeling.html"><strong aria-hidden="true">3.3.</strong> 序列建模</a></li><li class="chapter-item expanded "><a href="part3-ranking/multi-objective.html"><strong aria-hidden="true">3.4.</strong> 多目标优化</a></li><li class="chapter-item expanded "><a href="part3-ranking/multi-scenario.html"><strong aria-hidden="true">3.5.</strong> 多场景建模</a></li></ol></li><li class="chapter-item expanded "><a href="part4-rerank/index.html"><strong aria-hidden="true">4.</strong> 重排多样性建模</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part4-rerank/greedy-rerank.html"><strong aria-hidden="true">4.1.</strong> 基于贪心的重排</a></li><li class="chapter-item expanded "><a href="part4-rerank/personalized-rerank.html"><strong aria-hidden="true">4.2.</strong> 个性化重排</a></li></ol></li><li class="chapter-item expanded "><a href="part5-trends/index.html"><strong aria-hidden="true">5.</strong> 前沿趋势</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part5-trends/debiasing.html"><strong aria-hidden="true">5.1.</strong> 模型去偏</a></li><li class="chapter-item expanded "><a href="part5-trends/cold-start.html"><strong aria-hidden="true">5.2.</strong> 冷启动</a></li><li class="chapter-item expanded "><a href="part5-trends/generative-trend.html"><strong aria-hidden="true">5.3.</strong> 生成式范式演进</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">下篇 · 生成式推荐主线</li><li class="chapter-item expanded "><a href="part6-gr-basic/index.html"><strong aria-hidden="true">6.</strong> 生成式推荐范式基础</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part6-gr-basic/gr-paradigm.html"><strong aria-hidden="true">6.1.</strong> 生成式推荐范式</a></li><li class="chapter-item expanded "><a href="part6-gr-basic/gr-architecture.html"><strong aria-hidden="true">6.2.</strong> 生成式推荐架构基础</a></li><li class="chapter-item expanded "><a href="part6-gr-basic/llm-foundation.html"><strong aria-hidden="true">6.3.</strong> LLM 基础与推荐映射</a></li><li class="chapter-item expanded "><a href="part6-gr-basic/codebook-quantization.html"><strong aria-hidden="true">6.4.</strong> Codebook 量化与语义 ID</a></li></ol></li><li class="chapter-item expanded "><a href="part7-scaling/index.html"><strong aria-hidden="true">7.</strong> Scaling 生成式排序</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part7-scaling/hstu.html"><strong aria-hidden="true">7.1.</strong> HSTU：Scaling Law 的基石</a></li><li class="chapter-item expanded "><a href="part7-scaling/generative-ranking.html"><strong aria-hidden="true">7.2.</strong> 生成式排序范式</a></li><li class="chapter-item expanded "><a href="part7-scaling/mtgr.html"><strong aria-hidden="true">7.3.</strong> MTGR：混合范式</a></li><li class="chapter-item expanded "><a href="part7-scaling/rankmixer.html"><strong aria-hidden="true">7.4.</strong> RankMixer：硬件高效架构</a></li><li class="chapter-item expanded "><a href="part7-scaling/onetrans.html"><strong aria-hidden="true">7.5.</strong> OneTrans：统一 Transformer</a></li></ol></li><li class="chapter-item expanded "><a href="part8-e2e/index.html"><strong aria-hidden="true">8.</strong> 端到端生成式应用</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part8-e2e/e2e-recommendation.html"><strong aria-hidden="true">8.1.</strong> 端到端生成式推荐</a></li><li class="chapter-item expanded "><a href="part8-e2e/e2e-search.html"><strong aria-hidden="true">8.2.</strong> 端到端生成式搜索</a></li><li class="chapter-item expanded "><a href="part8-e2e/e2e-advertising.html"><strong aria-hidden="true">8.3.</strong> 端到端生成式广告</a></li></ol></li><li class="chapter-item expanded "><a href="part9-thinking/index.html"><strong aria-hidden="true">9.</strong> 推荐中的思考与推理</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part9-thinking/semantic-alignment.html"><strong aria-hidden="true">9.1.</strong> 协同语义与语言语义对齐</a></li><li class="chapter-item expanded "><a href="part9-thinking/reasoning-framework.html"><strong aria-hidden="true">9.2.</strong> 推理框架</a></li><li class="chapter-item expanded "><a href="part9-thinking/autonomous-reasoning.html"><strong aria-hidden="true">9.3.</strong> 自主推理探索</a></li></ol></li><li class="chapter-item expanded "><a href="part10-diffusion/index.html"><strong aria-hidden="true">10.</strong> 扩散模型推荐</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part10-diffusion/diffusion-basics.html"><strong aria-hidden="true">10.1.</strong> 扩散模型基础</a></li><li class="chapter-item expanded "><a href="part10-diffusion/diffusion-augmentation.html"><strong aria-hidden="true">10.2.</strong> 扩散做数据增强</a></li><li class="chapter-item expanded "><a href="part10-diffusion/diffusion-application.html"><strong aria-hidden="true">10.3.</strong> 扩散在推荐的应用</a></li></ol></li><li class="chapter-item expanded "><a href="part11-project/index.html"><strong aria-hidden="true">11.</strong> 生成式推荐系统实战</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part11-project/project-intro.html"><strong aria-hidden="true">11.1.</strong> 项目引言与目标</a></li><li class="chapter-item expanded "><a href="part11-project/project-architecture.html"><strong aria-hidden="true">11.2.</strong> 系统架构设计</a></li><li class="chapter-item expanded "><a href="part11-project/offline-pipeline.html"><strong aria-hidden="true">11.3.</strong> 离线管线</a></li><li class="chapter-item expanded "><a href="part11-project/online-pipeline.html"><strong aria-hidden="true">11.4.</strong> 在线管线</a></li><li class="chapter-item expanded "><a href="part11-project/frontend.html"><strong aria-hidden="true">11.5.</strong> 前端</a></li><li class="chapter-item expanded "><a href="part11-project/deployment.html"><strong aria-hidden="true">11.6.</strong> 部署上线</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">专题 · 计算广告</li><li class="chapter-item expanded "><a href="part12-computational-advertising/index.html"><strong aria-hidden="true">12.</strong> 计算广告</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="part12-computational-advertising/advertising-panorama.html"><strong aria-hidden="true">12.1.</strong> 计算广告全景与生态</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/billing-and-metrics.html"><strong aria-hidden="true">12.2.</strong> 计费模式与核心指标</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/auction-mechanisms.html"><strong aria-hidden="true">12.3.</strong> 竞价机制：从一价到二价</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/smart-bidding.html"><strong aria-hidden="true">12.4.</strong> 智能出价与预算控制</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/bias-and-calibration.html"><strong aria-hidden="true">12.5.</strong> 广告系统中的偏差与校准</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/open-closed-loop.html"><strong aria-hidden="true">12.6.</strong> 开环广告与闭环广告</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/online-allocation.html"><strong aria-hidden="true">12.7.</strong> 在线分配与流量管理</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/audience-targeting.html"><strong aria-hidden="true">12.8.</strong> 受众定向技术</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/ad-retrieval-semantic-recall.html"><strong aria-hidden="true">12.9.</strong> 广告检索与语义召回</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/data-processing-and-trading.html"><strong aria-hidden="true">12.10.</strong> 数据加工与交易</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/experiment-and-antifraud.html"><strong aria-hidden="true">12.11.</strong> 实验框架与反作弊</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/contract-advertising-products.html"><strong aria-hidden="true">12.12.</strong> 合约广告：产品形态与售卖模式</a></li><li class="chapter-item expanded "><a href="part12-computational-advertising/feed-native-ads.html"><strong aria-hidden="true">12.13.</strong> 信息流与原生广告</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">附录</li><li class="chapter-item expanded "><a href="GLOSSARY.html"><strong aria-hidden="true">13.</strong> 📖 术语表</a></li><li class="chapter-item expanded "><a href="appendix/word2vec.html"><strong aria-hidden="true">14.</strong> Word2Vec 专题附录</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
