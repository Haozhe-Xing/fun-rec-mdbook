<div style="display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; align-items: center;">
  <span style="background: #f0f4ff; color: #4A6CF7; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74,108,247,0.2);">📖</span>
  <span style="background: #fef9ec; color: #d97706; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(217,119,6,0.2);">⏱️ ~50 min read</span>
  <span style="background: #fef9ec; color: #d97706; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(100,100,100,0.15);">🎯 Intermediate</span>
</div>

# Feed and Native Advertising

> 📝 **Before You Continue:** Read 12.1 (ecosystem panorama) first — this chapter directly returns to the ad-form evolution ladder there; the section "feed ads: the positive exemplar of balancing performance and experience" is fully expanded here. Then 12.4 (smart bidding and budget control) — the eCPM formula of oCPC/oCPM and budget control are covered there from the algorithmic view; this chapter covers only the product side. The conversion-attribution chain of 12.6 (open/closed loop) is upstream of 12.13.4, and GSP/VCG from 12.3 (auction mechanisms) appears briefly.

Programmatic trading turned advertising into a business independent of content: traffic goes to exchanges, data to DMPs, decisions to DSPs — the relationship between ads and media content weakened. That was tolerable on the PC's large screen, but it hit a wall on mobile: small screens, imprecise touch interaction, and highly fragmented attention — one intrusive banner can ruin an entire reading experience. The industry's response was to **uniformly produce or jointly rank** commercial and non-commercial content — this is the direction of **native advertising (Native Ads)**, often called "content as ad." Strictly speaking, everything from advertorials to search ads to social feed ads reflects only one facet of native; but the most typical product form — and the earliest to trigger the debate — is the **feed ad**.

This chapter follows the route "challenge → core form → form spectrum → smart delivery → programmatic convergence": first why the mobile environment necessitated native (12.13.1's story), then the definition and mixing mechanism of feed ads (the chapter's core), then a tour of the native family — splash, interstitial, rewarded video — then the product view of how oCPX smart delivery lets small advertisers play the auction, and finally how native and programmatic trading — two seemingly opposite roads — converge.

After reading this chapter, you will be able to:

- State the mobile-era motivations for native advertising, plus the two new opportunities and core challenges of mobile versus PC advertising
- Define feed ads precisely with two key conditions and use them to classify whether a product form is "feed"
- Describe the mixing mechanism of feed ads: the multi-slot auction queue and the $S$/$K$ parameters' experience-revenue trade-off
- Distinguish expressive native from scene native, and explain why rewarded video is the highest-eCPM native form
- Describe the oCPX conversion-tracking chain and the three bid-expression modes from the product side, and work through 5 layered practice problems

---

## 12.13.0 Opening: Native Advertising Is the Fusion of Ad and Content Forms

First, an honest positioning of native advertising. **Native advertising** has no universally bulletproof definition — any product that uniformly produces or jointly ranks commercial and non-commercial content can be considered related to native. Advertorials are content produced for soft promotion; search ads appear in the same stream as organic results; social feed ads mix ads into the activity list. Each reflects one facet of "native," and the product philosophy that pushes this direction to its extreme is: an ad should not be something users must "tolerate" — it should be part of the content-consumption experience.

Why did this direction gain full attention only in the mobile internet era? Because displaying and operating ads independently of content met huge challenges on small screens. PC-era pages offered canvases over a thousand pixels wide where banners and skyscrapers each had their place; a mobile screen is a few inches, and touch interaction is far less precise than a mouse — a banner floating at the top of the page stays put while the user scrolls the content, and mis-clicks and annoyance happen simultaneously. The industry thus began exploring native advertising as a partial replacement for standard display ads to improve mobile monetization. Platform-level native products provided by third parties also emerged only in the mobile era.

> **Analysis:** In terms of product evolution, native advertising is not a rejection of programmatic trading but its completion. The 12.1 evolution ladder has two forces: mechanism evolution (from selling positions to selling audiences, from CPM to RTB) solves "selling efficiently"; form evolution (fusing content and ads) solves "users willing to look." The programmatic era pushed the former to its peak while the latter became the short board — ad trading independent of content necessarily hits ceilings in both performance and user experience. It is in this sense that native sits at the top of the evolution ladder.

---

## 12.13.1 Opportunities and Challenges of Mobile Advertising

The transaction forms of mobile advertising can be seen as a natural extension of PC internet advertising: display networks and search auctions were transplanted as-is to mobile; the transaction mechanisms and product forms of previous chapters still apply. But mobile has two distinct new opportunities. First, **the possibility of scenario-based advertising**: the mobile device never leaves the user's side, so location, life state, and intent can all be deeply understood — targeting can start from scenario and intent rather than just interest-based product pushing. For example, if location indicates the user is at work, game ads should not be pushed. Second, **a large pool of potential local advertisers**: even online, PC-era ads could only locate at city level — far too coarse for a neighborhood barbershop; mobile's GPS, cellular, and Wi-Fi positioning made local advertising feasible for the first time.

But the challenges behind the opportunities are equally concrete, concentrated in two points. First, **data fragmentation**: the mobile internet never formed a Web-centric ecosystem like the PC era; instead there is an app-centric system — apps are relatively isolated, there is no organizing system like hyperlinks, and data sources are fragmented and hard to integrate. Theoretically mobile knows the user better; in practice data acquisition is harder, and the data-exchange mechanisms common in the Web ecosystem largely fail in the app ecosystem. Second, **privacy and identity restrictions**: device identifiers and cross-app tracking keep shrinking under privacy regulation (ATT, Privacy Sandbox), further aggravating the data-side difficulty. These threads are covered in 12.6's identity infrastructure and 12.10's data processing and trading; this chapter does not repeat them.

The direct consequence: traditional banners misbehaved on mobile. Mobile banner CTR is far higher than PC banners, but much of it is mis-clicks — touch interaction is imprecise, mis-clicks severely disrupt the user's task and hurt experience; meanwhile advertisers observe poor conversion because most mis-clicks produce nothing. **Inflated CTR with relatively poor conversion** — this combination says mobile needs a new creative and product approach, which is the biggest driver of going native.

---

## 12.13.2 Feed Ads: The Mixing Problem and the Experience-Revenue Balance

Feed ads first appeared in social networks and were later widely adopted by all kinds of mobile ad products. Their effectiveness, in formal terms, comes from two things: interaction coupling with content, and relative independence from surrounding content. Based on extensive product practice, a descriptive definition:

> A **feed ad** is an ad form such that: first, the ad interacts in a manner coupled with the content; second, the segments of content separated by the ad have no direct relation to each other.

The first condition — "interaction coupling" — means: when the user scrolls through content, the embedded ad is operated the same way — however you operate content, that's how you operate the ad. This has two benefits: operation becomes far more convenient and mis-clicks drop; and the user perceives the ad as part of content consumption, raising attention and ad effect. Counter-examples are easy to find: a traditional banner floating motionless while content scrolls fails this condition; an interstitial closed by a corner button is not a typical content interaction either — but if the ad is dismissed by swiping and the user then enters other content, it can be classified as a feed ad. The second condition — "content relatively independent" — means: the content blocks separated by the ad each independently express one item, with no continuation or causal relation. If the blocks are strongly related, users perceive the ad as interrupting their current reading task — harming both ad attention and product experience. So an ad slot carved into the middle of an article is not a feed ad; whereas social networks and news clients have naturally weakly-linked content blocks, ideal for feed monetization. As for "display style consistent with content blocks" and "precise audience targeting" — common in practice, but not fundamental features, so they stay out of the definition.

### 🧠 Mental Model: Buffet Tray vs. Serialized Novel

> Think of a feed as a buffet tray and a long article as a serialized novel. In the tray, each compartment's food is unrelated to the next — mixing in a few "sponsored dishes" is barely noticeable, and you pick them up with the same motion. Inserting an ad page mid-novel interrupts the reader at the plot's climax — only anger. The two definition conditions say exactly this: the ad must be picked up the same way as the tray (interaction coupling), and the tray's compartments are inherently independent (unrelated content). Ads inside a serial? That's another business and needs another approach.

### Multi-Slot Auctions and S/K: The Product Mechanism of Mixing

In product essence, feed ads differ little from ordinary display advertising — they can be viewed as a **multi-slot auction product with fairly free ad placement**. Their display and interaction form belongs to another category: the feed ecosystem likewise has closed ADNs, supply-side ADX and SSP, and demand-side DSPs, with the same core decision flow as display. What is genuinely distinctive are two product questions.

![The mixing mechanism of feed ads: organic content and ad cards compete for positions in one stream; S and K control ad density](../images/part12-native-feed-mix.svg)

In the figure, the several ad slots $a_1, a_2, a_3, \ldots$ inserted into the feed form an **auction queue**, chargeable by VCG or GSP; one can also treat $a_1, a_2$ as different slots auctioned separately, but if de-duplication across these slots is required, that is equivalent to a single auction queue. The second question is subtler — **where ads appear**. It is governed by two parameters: $S$, the position (after how many content items) of the first ad; and $K$, the gap (number of content items) between consecutive ads. The larger $S$ and $K$, the less user attention ads receive and the smaller the impact on user experience. This is isomorphic to the ad-placement problem in search: under an **average ad count** constraint, tune each user's $S$ and $K$ to optimize total ad CTR — the average-count constraint acts as the user-experience constraint, and the key to solving it is accurately estimating each user's CTR level relative to the population.

### Natural Isomorphism with Recommender Systems: Mixing

Now zoom out. In feeds today, organic content is ranked by relevance and ads by eCPM; the two come from different services with different criteria, then mixed by fixed logic — this is how most search and feed products work. But the ultimate direction of "content as ad" is: content and ads **ranked under one unified criterion**, so every position allocation is decided by a unified score competition. This is exactly the **mixing (re-ranking / mixing)** problem of recommender engineering: ads and organic content as two kinds of items in one candidate pool, competing for display positions with comparable scores (organic content's experience value vs. ads' eCPM and experience cost). For recommender-system engineers, this is the most direct engineering junction of advertising and recommendation — your familiar toolbox of diversity, experience modeling, and multi-objective fusion scores is precisely what solves mixing.

> **Analysis:** Feed ads are called the exemplar of balancing performance and experience because they capture both the formal gain of "ads close to content" and the mechanism gain of "auctions preserving efficiency": formally, interaction coupling and content independence make users attend to ads naturally; mechanically, the multi-slot auction queue retains the price-clearing efficiency of 12.3. The cost is two new constraints — creatives must fit all feed positions (the "expressive native" problem of 12.13.3), and density must be finely balanced via $S$/$K$. One product detail worth noting: any platform, after fully monetizing its own traffic, expands to off-platform inventory, and off-platform products are complicated — without native rendering, fill rates collapse. So even Facebook-like single large-traffic platforms cannot escape the adaptation problem.

---

## 12.13.3 The Native Form Spectrum: From Splash to Rewarded Video

Beyond feeds, mobile advertising has spawned a family of creative forms deeply integrated with content. Ordered by "degree of nativeness," they form a spectrum.

**Patches on traditional forms: banner and interstitial.** Banners come straight from PC, with mobile's inflated-CTR/poor-conversion problem (analyzed in 12.13.1). **Interstitial ads** resemble pause ads in video, appearing when a game or app pauses — likewise inflated CTR, relatively poor conversion. But thanks to mature trading systems (networks, exchanges), these highly standardized forms scale most easily and remain the mainstay of mobile display, sold mostly by auction.

**Forms going with the flow: splash and lock screen.** **Splash ads** display a full-screen ad on the app's loading page. This is a rather good mobile form exploration: while waiting for the app to open, the user has no active task, so annoyance is low; full-screen display also carries more brand value, and selling is often **contract-based**. **Lock-screen ads** display when the device is locked; similar properties to splash, low experience impact, but mostly incentive-based.

**Incentivized download forms: offerwalls and points walls.** Budgets targeting app downloads spawned dedicated forms. **Offerwalls** push download ads directly, analogous to off-platform recommendation; **points walls** grant points redeemable for virtual goods after download and activation. They belong to incentive advertising alongside rebate sites — clicks and activations look good, but downstream retention is poor. Yet they have unique value in special scenarios: chart-climbing launches need mass downloads fast; new game servers need players to gather quickly — both once relied on points walls. But Apple explicitly cracks down on using incentives to influence charts, so their prospects are dim.

### Rewarded Video: Why It Is the Best-Performing Native Form

**Rewarded video advertising** is the most representative product of the native direction, common in game media, with a four-step flow:

1. Natural in-game entry — when the player is stuck or wants a virtual item, a prompt offers a video view in exchange for a reward;
2. The user opens the video; a 15–30 second ad plays and cannot be skipped;
3. On completion, a download or other conversion landing page is shown;
4. The user returns to the game and receives the virtual-item reward.

Its performance advantage comes from two sources. First, in the incentive scenario the user cannot skip and must watch to the end — having received the full message, conversion naturally improves; second, the in-game virtual good bundled with the ad is scarce for non-paying users, making them willing to watch attentively even when the real-world value is low. This differs essentially from points-wall incentives: rewarded video **rewards only the viewing behavior and does not stimulate downloads**, so it does not suffer the low-user-quality problem. Precisely for this reason, rewarded-video networks serve performance advertisers better than brands — mainstream rewarded-video networks derive most revenue from app-download ads. Commercial results corroborate this: AppLovin, with rewarded video at its core, founded in 2012, had exceeded $500M revenue and $90M net profit by 2016.

> **Analysis:** Rewarded video is an exquisite balance of native logic: it requires natural in-game entry and seamless integration with the game's points system — the media pays a design cost; but its core — video creative playback — is a highly standardized process, very easy to trade programmatically. "Deeply customized scenario shell + standardized tradeable kernel" — this is a general recipe for scaling native products, and it explains why rewarded video became a major direction of mobile advertising.

### Native Ad Platforms: Expressive Native and Scene Native

Pull the view back from creative forms to the platform layer. "Native" actually contains two different aspirations: making the ad's display style and format consistent with the content — **expressive native**; and keeping the ad's targeting decision logic consistent with content production, triggered by user scenario — **scene native**. Against earlier examples: social feeds lean expressive native; search ads are native in both senses — their delivery decisions follow exactly the display principles of content results, matching ads the way content is matched. From this we can summarize two product principles for native platforms: expressive native requires **the media to control the ad display form** (even fonts and colors must adapt to the media — a demand traditional "creatives" cannot carry); scene native requires **using the media's scenarios and needs to filter ads**.

The ideal native platform combines both and operates at scale as a third party. The mechanism is called **embedded native advertising**: after the media judges user scenario and intent, it requests **structured paid content** from the ad platform via a structured query (e.g., "type=hotel; location=Lhasa") — the platform returns not finished creatives but assemblable field-level material that the media renders in its own style template. This goes beyond contextual targeting: contextual targeting has the ad platform guess page topics with shallow NLP, whereas the media's active participation makes intent extraction far easier. The challenges are real too: media participation adds degrees of freedom and greatly raises operating complexity; onboarding small and mid-size media needs long market cultivation; and accumulating structured, per-industry paid-content libraries takes time — even large platforms hold creatives at scale, not content libraries.

---

## 12.13.4 oCPX Smart Delivery: The Product View

Mobile advertising has one more important product difference from the PC era: the wide adoption of **smart delivery (oCPX)**. This section covers only the product side — how advertisers use it and how the chain is wired; the eCPM formula of oCPC/oCPM and bid scaling under budget constraints are fully covered from the algorithmic view in 12.4, cross-linked here without repetition.

### Why oCPX: Dismantling the Barrier for Small Clients

The basic problem of smart delivery is clear: the platform takes on more computation during bidding to help small and mid-size clients with limited data and IT capability, lowering both the comprehension barrier and the optimization cost of auction advertising. But there is a trap: pure CPA/CPS/ROI billing would attract a flood of problematic clients free-riding traffic — bad money driving out good. Mainstream products' solution is the **oCPX model separating billing from bidding**: billing still runs the old CPM/CPC path, but the advertiser's expressed goal becomes conversion cost — the platform trades "I optimize for your conversions" for "billing conventions unchanged."

![The product chain of oCPX smart delivery: the advertiser bids on conversions, the platform estimates and auctions, conversion tracking feeds back, and bids adjust to actual cost](../images/part12-native-ocpx-flow.svg)

The chain, taken apart from the product view, is four steps: the advertiser sets a conversion bid (value per conversion) and budget; the platform estimates CTR and CVR and enters the ranking auction with $\mathrm{eCPM} = \mu(a,u,c) \cdot c(a,u) \cdot \mathrm{bid}_{\mathrm{CPA}}(a)$ (formula derivation and budget control in 12.4.1 and 12.4.2); the user clicks, downloads, converts on the media; conversion events are attributed and reported back to the platform (attribution conventions and ATT/SKAN limitations in 12.6; anti-fraud in 12.11). The loop's last link: the platform tracks the deviation of actual conversion cost from the bid expectation and dynamically adjusts the true bid.

### CVR Estimation: Why Mobile Made It Work

From the platform's technical view, the new problem oCPX introduces is mainly **conversion-rate estimation**. It is far harder than CTR estimation: conversion funnels differ greatly across industries, defeating unified modeling; and conversion data are much sparser than clicks. In the PC era, these two mountains kept CVR estimation promising on paper but impractical.

The turning point came from an unglamorous mobile fact: **conversion funnels became far more consistent**. In mobile advertising, many industries' conversions take the form of app downloads — e-commerce, gaming, utility, and finance campaigns often start with getting the client's app. From a technical (not commercial) view, there is only one real conversion outlet: the app store. All download-type clients share one data pipeline, so they can in principle be modeled jointly, greatly easing sparsity. Inspired by this, platforms keep pushing funnel unification beyond app downloads — on Chinese platforms, for example, independent e-commerce sites gradually gave way to the platform's unified site templates, one purpose of which is that unified funnels aid CVR modeling.

### Three Understandings of Bidding: Bid, Price, and Budget

oCPX has a hidden but crucial product question: should the platform interpret the advertiser's conversion bid $\mathrm{bid}_{\mathrm{CPA}}(a)$ as a **bid** or as the **actual conversion cost the advertiser expects to pay (price)**? The two understandings correspond to entirely different market mechanisms.

Interpreted as a bid, the platform runs second-price: rank by $\mathrm{eCPM} = \mu \cdot c \cdot \mathrm{bid}_{\mathrm{CPA}}$, and charge the winner down to the next eCPM. Billing is still CPM, but the client's actual CPA cost can be back-computed — with reliable estimates and sufficient budget, this cost is necessarily below the bid; the market is essentially no different from a second-price CPC market, and truthfulness and social-welfare optimality are preserved. The client only needs to know roughly the value of each conversion and bid truthfully. Interpreted as a price, the market is effectively first-price: since CTR and CVR estimates are necessarily biased, naive first-price handling cannot keep actual conversion cost near the bid, so the platform must additionally **track actual conversion cost and dynamically adjust the true bid according to its deviation from expectation**, compensating until cost converges to the bid. Facebook's product terms map exactly onto the two understandings: the former is close to **bid cap**, the latter close to **cost cap** or **target cost**.

For most clients, even conversion bidding is too complex — what concept does everyone understand? Only budget. Facebook pioneered **budget expression**: the client states only a daily spend; the platform smooths it across time slots — still within the auction framework: the platform forecasts each slot's market price for the chosen audience and back-computes the bid needed to spend on plan. This "foolproof" bidding — make creatives, pick audience, set budget, press start — greatly increased the number of active market clients. But time-slot back-computation violates the truthfulness of second-price markets in a sense: when the back-computed CPA bid exceeds the client's true conversion value in some slot, losses can occur — hence Facebook kept bid cap for capable clients, substituting their cap when it is below the system's back-computed bid.

> **Analysis:** The three bid expressions form a "finesse vs. comprehension barrier" ladder: bid cap is finest and preserves second-price mechanism properties, but requires understanding "bid"; cost cap / target cost turns the promise into "actual cost" — cheaper to teach, but the market degrades to first-price and needs platform-side compensation; budget expression has the lowest barrier, but back-computed bids can overpay when they deviate from the client's true value. The author's view is worth savoring: gradually guiding the market and clients from the second-price basis may be the reasonable path — mechanism health and barrier reduction must, in the long run, both be kept.

---

## 12.13.5 Closing: The Convergence of Native Advertising and Programmatic Trading

Finally, answer a seemingly contradictory question: programmatic trading trends toward audience buying and automated auctions, while native advertising demands deep media participation and fusing ads into content — are these two roads opposed? Where do they converge?

First observe: could search advertising ever be programmatically traded? No such product has ever been seen in the market. Yet in Facebook's feed ads there is delivery by advertiser-uploaded audience lists — not programmatic trading, but similar in purpose, and easily convertible to RTB. Both are special forms of native advertising; why such different acceptance of programmatic trading? The key: **whether native ads are triggered by user intent**. In native advertising with explicit user intent (like search), fully open RTB makes relevance of paid results hard to control — only a few technically strong platforms can achieve good relevance; letting many DSPs into the auction cannot guarantee result quality, so a single strong native network (or self-operation) is more viable. In native forms like social feeds, where user intent is not explicit and ads are not required to be intent-triggered, programmatic trading is entirely viable — and this is one of native advertising's future trends.

Placing this conclusion back on the 12.1 evolution ladder, this chapter's position is clear: the ladder is driven by form evolution (content-ad fusion) and mechanism evolution (transaction automation), converging at the top. Native RTB is exactly that convergence point — "personalized content" (native makes ads appear in ways users accept) and "programmatic transactions" (RTB prices every personalized decision in an open market) are no longer a contradiction but two sides of the same decision system. The row "feed ads: beginning to fuse content and ads" in 12.1's evolution ladder only truly lands in this chapter.

---

## ⚠️ Common Mistakes in 12.13

| # | Mistake | Example | Why It's Wrong | Fix |
|---|---------|---------|---------------|-----|
| 1 | Understanding feed ads as "just look like content" | An ad slot inside article paragraphs styled like body text, calling itself feed | The second defining condition requires the content separated by the ad be mutually independent; long-article blocks are continuous, and insertion interrupts the reading task | Judge by both defining conditions: interaction coupling + unrelated content blocks, neither dispensable |
| 2 | Confusing oCPX's three bid expressions | A client bids with cost cap but expects bid-cap second-price charging and complains of overcharging | Under bid cap the bid is a bid (second-price market); under cost cap it is an expected cost (first-price + platform adjustment); budget mode needs no bid at all | Confirm the mode before onboarding; explain the expected-cost convention and the adjustment mechanism |
| 3 | Believing higher ad density means higher revenue | Shrinking $S$ and $K$ relentlessly for per-session revenue | High density brings mis-clicks and annoyance, damaging long-term traffic value; and a feed is an auction queue — bad positions drag overall CTR | Tune $S$/$K$ per user segment under an average-ad-count constraint, optimizing overall CTR |
| 4 | Equating rewarded video with points-wall incentives | Worrying rewarded video "brings low-quality users" | Rewarded video rewards viewing only and does not stimulate downloads, avoiding the points-wall quality problem | Distinguish "reward viewing" from "reward downloading"; evaluation criteria differ accordingly |
| 5 | Assuming intent-based native ads can open RTB directly | Wiring search ad slots into an open auction | For native ads triggered by explicit user intent, open bidding cannot control paid-result relevance | Intent-explicit native goes to a strong single-platform network or self-operation; intent-ambiguous feed-type native suits programmatic |
| 6 | Blaming only the model when conversion estimates are off | Iterating models endlessly on high pCVR bias, ignoring the tracking chain | Conversion events depend on reporting and attribution conventions; chain data loss or attribution mismatch corrupts model inputs | First verify reporting completeness and conventions along the 12.6 attribution chain, then diagnose the model |

---

## Chapter Summary

### 📌 Key Takeaways

| Concept | Key Points | Why It Matters |
|---------|-----------|----------------|
| Why native | Small mobile screens, imprecise interaction, banner mis-clicks with poor conversion; produce/rank commercial and non-commercial content together (content as ad) | Explains why going native became a mobile-era necessity, not a style choice |
| Feed ad definition | Two conditions: ad interacts coupled with content; content separated by the ad is mutually independent | The only standard for classifying product forms; similar styling is not sufficient |
| Mixing & placement | Multiple ad slots form an auction queue (VCG/GSP); $S$ (first position) and $K$ (gap) set density; constrain average ad count, optimize overall CTR | The core product mechanism of feeds, and the engineering starting point of recommender "mixing" |
| The native spectrum | Expressive native (media control display style) vs. scene native (scenario/intent triggering); rewarded video = deeply customized scenario + standardized tradeable kernel, rewarding viewing not downloads | Native is not one form but a spectrum ordered by degree of nativeness |
| oCPX product chain | Billing separated from bidding; mobile unified conversion funnels via app stores easing sparsity; three bid expressions: bid cap (second-price) / cost cap (first-price + compensation) / budget (back-computed bids) | The product infrastructure letting small clients join auctions; mechanism properties vary with expression mode |
| Native–programmatic convergence | Whether triggering depends on explicit user intent determines RTB openness; intent-ambiguous feed-type native can combine with programmatic trading | Returns to 12.1's ladder: form evolution and mechanism evolution converge at the top |

### ❓ FAQ

**Q1: How do feed ads really differ from ordinary display ads, if both are auctions underneath?**
> The decision pipeline (retrieval, ranking, auction, billing) is essentially identical; the feed ecosystem likewise has ADN/ADX/DSP. The differences concentrate in display and interaction: interaction must couple with content, and content blocks must be mutually independent — from which two distinctive product problems derive: creative adaptation and $S$/$K$ density control. In one sentence: the same auction kernel, wrapped in a "native" shell with new constraints.

**Q2: Under budget expression, can the platform's back-computed bids lose my money?**
> Yes. When the back-computed CPA bid in some slot exceeds your true conversion value, you overpay — exactly the price budget expression pays for violating truthfulness. Clients with operating capability should use bid cap: when the system's back-computed bid exceeds your cap, your cap governs.

**Q3: Why is it "counterintuitive" that rewarded video serves performance advertisers?**
> Intuitively, full-screen video deeply embedded in game scenes looks like brand territory. But the two properties of the incentive scenario — must-watch-to-end, and virtual goods scarce for non-payers — naturally favor the "watch-then-convert" performance funnel, so mainstream rewarded-video networks earn mostly from app-download ads. Brand needs fit better with contract-sold full-screen forms like splash.

### 🔗 Connections

- **12.1** (ecosystem panorama): this chapter fully expands the "feed/embedded native" rows of 12.1's evolution ladder; native RTB is the concrete form of the ladder-top convergence of form and mechanism evolution
- **12.4** (smart bidding & budget control): the oCPC/oCPM eCPM formula, budget pacing, and bid scaling from the algorithmic view are in 12.4; this chapter covers only the product-side chain and bid expressions
- **12.6** (open/closed loop advertising): the attribution system oCPX's conversion tracking depends on, and the ATT/SKAN shock to mobile conversion, are in 12.6
- **12.3** (auction mechanisms): VCG/GSP charging of the feed's multi-slot queue, and the theoretical basis of second-price truthfulness
- **12.11** (experimentation & anti-fraud): risks of oCPX conversion data being polluted by attribution fraud, and detection methods, are in 12.11

---

## Practice Problems

Work through all problems in order — they get progressively harder. Each has a complete solution you can reveal after trying it yourself.

---

**Problem 12.13.1 — Classifying Feed Ads by the Two Defining Conditions** 🟢 Easy

Using the definition of 12.13.2 (interaction coupling + content separated by the ad mutually independent), judge whether each of the following four forms is a feed ad, with reasons:

(a) A card ad inside a social network activity list, scrolling with the content;
(b) A fixed banner between article paragraphs, floating motionless while content scrolls;
(c) An interstitial popping up on game pause, closed by a corner button;
(d) An in-app ad dismissed by swiping, after which other content follows.

**Sample Input:** `[(a), (b), (c), (d)]`
**Sample Output:** `(a) yes; (b) no; (c) no; (d) yes`
<details>
<summary>💡 Solution (click to reveal)</summary>
**Approach:** Check each against the two conditions: interaction coupling (operate the ad as you operate content) and mutually independent content blocks (no continuation or causality).

- (a) Scrolling with content satisfies coupling; activity-list blocks are naturally independent. Both hold — **yes**.
- (b) The floating banner violates coupling; and one article's blocks are continuous, violating independence. Both fail — **no**.
- (c) A corner-button close is not a typical content interaction; violates coupling — **no**.
- (d) Dismissal by swiping is an interaction consistent with content; can be classified as a feed ad — **yes**.

**Key points:**
- "Style consistent with content" and "precise targeting" are common but not fundamental features; they do not enter the judgment
- Judge only interaction and content structure, not whether the ad looks like content
</details>

---

**Problem 12.13.2 — eCPM Ranking and Actual CPA in the oCPX Second-Price Market** 🟢 Easy

Two candidate ads (mobile download-type) in a feed auction queue:

- Ad A: $\mu = 2\%$ (CTR), $c = 5\%$ (CVR), conversion bid $\mathrm{bid}_{\mathrm{CPA}} = ¥40$
- Ad B: $\mu = 1.5\%$, $c = 4\%$, $\mathrm{bid}_{\mathrm{CPA}} = ¥50$

Rank by eCPM to determine the winner; under second-price logic (charging down to the next eCPM), compute the winner's actual CPA cost and compare with A's bid.

**Sample Input:** $\mu_A = 0.02,\ c_A = 0.05,\ \mathrm{bid}_A = 40$; $\mu_B = 0.015,\ c_B = 0.04,\ \mathrm{bid}_B = 50$
**Sample Output:** A wins; $\mathrm{eCPM}_A = ¥40$, $\mathrm{eCPM}_B = ¥30$; actual CPA $= ¥30 < ¥40$
<details>
<summary>💡 Solution (click to reveal)</summary>
**Approach:** Rank by the 12.4.1 formula $\mathrm{eCPM} = \mu \cdot c \cdot \mathrm{bid}_{\mathrm{CPA}} \times 1000$; under second-price charging, convert CPA as "revenue per 1000 impressions ÷ conversions per 1000 impressions."

- $\mathrm{eCPM}_A = 1000 \times 0.02 \times 0.05 \times 40 = ¥40$; $\mathrm{eCPM}_B = 1000 \times 0.015 \times 0.04 \times 50 = ¥30$. A wins.
- Second price charges ¥30 per 1000 impressions. Conversions per 1000 impressions $= 1000 \times 0.02 \times 0.05 = 1$.
- Actual $\mathrm{CPA} = 30 / 1 = ¥30 < ¥40 =$ bid.

This is exactly the second-price property of 12.13.4: with reliable CTR/CVR estimates and sufficient budget, actual CPA is necessarily below the bid, and the market preserves truthfulness — the client just reports each conversion's value honestly.
**Key points:**
- oCPX billing is still CPM; CPA is back-computed from second-price charges
- Actual CPA below bid is not luck but a second-price necessity (here 30 < 40)
</details>

---

**Problem 12.13.3 — Feed Density Placement with S/K** 🟡 Medium

A news app loads 30 content items per refresh; product requires **no more than 3 ads on average per refresh**. If the first ad appears fixed after the 3rd item ($S = 3$), and the next two ads follow at the same interval $K$ (after items $S + K$ and $S + 2K$), find the maximum integer $K$ satisfying the constraint, give the three ads' positions, and verify that $K + 1$ breaks the limit.

**Sample Input:** total content 30; ad cap 3; $S = 3$
**Sample Output:** $K = 13$; ads after items 3, 16, and 29
<details>
<summary>💡 Solution (click to reveal)</summary>
**Approach:** The third ad's position is $S + 2K$ and must not exceed 30: $3 + 2K \le 30$.

- $2K \le 27 \Rightarrow K \le 13.5$, so the maximum integer is $K = 13$.
- Positions: after items 3, $3+13=16$, $16+13=29$ — all within 30.
- Verify $K = 14$: $3 + 2 \times 14 = 31 > 30$ — the third ad overflows; infeasible.

Note the trade-off meaning of $S$/$K$: under the "at most 3" cap, $K$ at its maximum means ads are diluted toward the tail — less attention, less experience harm; monetizing the first half requires shrinking $K$ and accepting the cost. 12.13.2's "fine-tune per user's CTR level" is optimization exactly on this boundary.
**Key points:**
- The constraint is $S + (n-1)K \le$ total content, $n$ being the number of ads
- The cap constrains only "average ad count"; positions ($S$, $K$) are the actual degrees of freedom of the experience-revenue trade-off
</details>

---

**Problem 12.13.4 — eCPM Comparison: Banner vs. Rewarded Video** 🔴 Hard

A game media has two slot options; the advertiser's CPA bid $\mathrm{bid}_{\mathrm{CPA}} = ¥20$ is unchanged:

- Option 1 (banner): CTR $\mu = 0.5\%$, post-click conversion $c = 1\%$;
- Option 2 (rewarded video): unskippable, must-watch; CTR $\mu = 15\%$, post-watch conversion $c = 2\%$.

Compute both options' eCPM and the ratio; combined with rewarded video's two performance sources (must-watch, virtual-good-incentivized attentive viewing), explain why mainstream rewarded-video networks earn mainly from performance advertisers.

**Sample Input:** $\mathrm{bid} = ¥20$; $(\mu_1, c_1) = (0.005, 0.01)$; $(\mu_2, c_2) = (0.15, 0.02)$
**Sample Output:** $\mathrm{eCPM}_1 = ¥1$, $\mathrm{eCPM}_2 = ¥60$, 60×
<details>
<summary>💡 Solution (click to reveal)</summary>
**Approach:** Apply $\mathrm{eCPM} = \mu \cdot c \cdot \mathrm{bid}_{\mathrm{CPA}} \times 1000$ directly; compare $\mu \cdot c$ (conversions per 1000 impressions).

- Option 1: $1000 \times 0.005 \times 0.01 \times 20 = ¥1$. Conversions per 1000 impressions $= 0.05$.
- Option 2: $1000 \times 0.15 \times 0.02 \times 20 = ¥60$. Conversions per 1000 impressions $= 3$.
- Ratio: $60 / 1 = 60\times$, entirely from $\mu \cdot c$ rising 0.05 → 3 (60×).

Decomposed: CTR rises 30× (0.5% → 15%; unskippable + reward-on-completion guarantees the watch-and-click flow), CVR rises 2× (1% → 2%; full information transfer improves downstream conversion). The two performance sources of 12.13.3 map to these: must-watch powers CTR; attentive, incentive-driven viewing powers CVR. The large deterministic CVR gain lets performance advertisers (mostly app downloads) pay high CPAs for rewarded video — hence the networks' revenue structure.
**Key points:**
- eCPM gaps decompose fully onto the $\mu$ and $c$ factors; the ratio is invariant at fixed bids
- "Unskippable" is rewarded video's core product lever: it reshapes both funnel stages, exposure-click and click-conversion
</details>

---

**Problem 12.13.5 — Designing a Diagnosis for CPA Overruns** 🏆 Challenge

You lead an oCPX product at a mobile ad platform. Operations reports: a batch of cost-cap clients complain of actual conversion costs 30%+ above bid expectations, and bid adjustment fails to converge. Design a diagnosis: list at least 4 candidate root causes (from tracking chain, attribution conventions, model, and mechanism respectively), the observable metric and verification method for each, and the fix.

**Sample Input:** client complaint list + platform impression/click/conversion logs + reporting data + bid-adjustment records
**Sample Output:** a root-cause × metric × verification × fix table
<details>
<summary>💡 Solution (click to reveal)</summary>
**Approach:** Walk the oCPX chain (steps ①→④ of 12.13.4) from data source to mechanism end: first decide "is the cost really high, or is it a data-convention mismatch," then separate "model estimation error" from "mechanism non-convergence."

| Candidate root cause | Observable metric | Verification | Fix |
|---------|-----------|---------|---------|
| Tracking-chain loss/latency | Daily reconciliation of reported conversions vs. app-store-side activations; reporting-latency distribution | Sample device-level comparison of the full impression-click-report chain | Fix reporting retries; late-conversion attribution compensation |
| Attribution-convention mismatch | Divergence stats between platform and client/MMP conventions (post-click window, last-click vs. multi-touch) | Recompute CPA for the same conversions under both conventions | Align windows and conventions with clients (see 12.6); display conventions explicitly in-product where alignment is impossible |
| pCVR model bias | Per-industry calibration curves of estimated vs. actual CVR (bucketed, cf. 12.5) | Plot calibration curves by industry and audience bucket; look for systematic over/under-estimation | Per-industry modeling or industry features; feed estimation bias into the bid-adjustment prior |
| Mechanism convergence failure | Adjustment response speed vs. cost volatility; extreme-slot back-computed bid records | Replay bid-adjustment trajectories in high-volatility periods (e.g., promotions); check lag or overshoot | Adaptive step sizes; bid guard-bands in volatile slots |
| Client misunderstanding (control) | Mismatch between the client's actual mode and expectation; spend curves | Check client configs: budget-expression mode mistaken for cost-cap expectations | Client education on the three expressions' (bid cap / cost cap / budget) expected-cost conventions |

**Key points:**
- First bisect "data wrong" vs. "mechanism wrong": chain and attribution problems corrupt model inputs, and no model can fix that
- cost cap is first-price + compensation; convergence depends on the signal-to-noise of "track actual cost → adjust bid"; data pollution drags it down directly
- Don't skip the control row: a sizable share of "complaints" stem from misunderstanding bid-expression modes — itself the product problem 12.13.4 emphasizes
</details>
