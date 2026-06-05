import { C, addArrow, addBg, addFooter, addKicker, addLabel, addMetric, addStep, addTitle } from "./common.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addKicker(slide, ctx, "THESIS", "01");
  addTitle(
    slide,
    ctx,
    "BGE-M3 learned concat-PCA 是当前最强的 point-estimation embedding",
    "核心不是替换 BGE-M3，而是在保留 BGE-M3 主信号的基础上加入 reward-informed correction。"
  );

  addMetric(slide, ctx, {
    x: 60,
    y: 220,
    w: 260,
    label: "BEST MIXED / SONNET",
    value: "RMSE 0.0674",
    note: "BGE-M3 learned concat-PCA + MIPS",
    accent: C.teal,
  });
  addMetric(slide, ctx, {
    x: 340,
    y: 220,
    w: 260,
    label: "RELATIVE BIAS",
    value: "-8.34%",
    note: "baseline BGE-M3 MIPS was -8.97%",
    accent: C.gold,
  });
  addMetric(slide, ctx, {
    x: 620,
    y: 220,
    w: 260,
    label: "DIRECTION",
    value: "100%",
    note: "all bootstrap samples recover agent > clinician",
    accent: C.blue,
  });
  addMetric(slide, ctx, {
    x: 900,
    y: 220,
    w: 260,
    label: "LIMITATION",
    value: "CI miss",
    note: "upper CI 0.7994 < truth 0.8413",
    accent: C.red,
  });

  addStep(slide, ctx, {
    x: 80,
    y: 410,
    w: 270,
    h: 120,
    title: "1. Frozen BGE-M3",
    body: "Text is first embedded by BGE-M3 into 1024-dimensional vectors for context and actions.",
    accent: C.blue,
  });
  addArrow(slide, ctx, 360, 468, 90);
  addStep(slide, ctx, {
    x: 470,
    y: 410,
    w: 270,
    h: 120,
    title: "2. Conservative learned layer",
    body: "Use PCA(BGE) to keep geometry and learned projection to add reward-informed signal.",
    accent: C.teal,
  });
  addArrow(slide, ctx, 750, 468, 90);
  addStep(slide, ctx, {
    x: 860,
    y: 410,
    w: 270,
    h: 120,
    title: "3. Re-run estimators",
    body: "Use the same [context; action; interaction] features for DM, MIPS and OffCEM.",
    accent: C.red,
  });

  addLabel(
    slide,
    ctx,
    "Takeaway: learned embedding helps only when it is used as a supplement to BGE-M3, not as a full replacement.",
    90,
    585,
    1050,
    38,
    { size: 19, bold: true, color: C.ink, align: "center" }
  );
  addFooter(slide, ctx);
  return slide;
}
