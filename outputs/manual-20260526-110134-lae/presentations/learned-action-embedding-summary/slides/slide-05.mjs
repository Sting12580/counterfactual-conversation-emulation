import { C, addArrow, addBg, addBox, addFooter, addKicker, addLabel, addStep, addTitle } from "./common.mjs";

function tower(slide, ctx, x, title, input, output, accent) {
  addStep(slide, ctx, {
    x,
    y: 235,
    w: 230,
    h: 92,
    title,
    body: `${input}\n1024-d BGE-M3 vector`,
    accent,
  });
  addArrow(slide, ctx, x + 75, 350, 70, accent);
  addStep(slide, ctx, {
    x,
    y: 380,
    w: 230,
    h: 112,
    title: "Projector MLP",
    body: "Linear → ReLU → Linear → Tanh\nL2 normalize after projection",
    accent,
  });
  addArrow(slide, ctx, x + 75, 515, 70, accent);
  addStep(slide, ctx, {
    x,
    y: 548,
    w: 230,
    h: 70,
    title: output,
    body: "learned latent representation",
    accent,
  });
}

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addKicker(slide, ctx, "MODEL DETAIL", "05");
  addTitle(
    slide,
    ctx,
    "Projection 模型是一个 two-tower reward predictor，只用 clinician reward 训练",
    "它学习 context projector 和 action projector；训练完成后，同一个 action projector 被应用到 agent action。"
  );

  tower(slide, ctx, 70, "Context tower", "phi_x", "z_x", C.blue);
  tower(slide, ctx, 350, "Action tower", "phi_a_clinician", "z_a_clinician", C.teal);

  addBox(slide, ctx, { x: 660, y: 245, w: 260, h: 120, fill: "#FFFFFF", stroke: C.faint });
  addLabel(slide, ctx, "Reward head input", 682, 266, 200, 24, { size: 15, bold: true });
  addLabel(slide, ctx, "[z_x ; z_a_clinician ; z_x * z_a_clinician]", 682, 304, 220, 36, {
    size: 12,
    color: C.muted,
  });
  addArrow(slide, ctx, 754, 388, 70, C.gold);
  addBox(slide, ctx, { x: 660, y: 420, w: 260, h: 118, fill: "#FFFFFF", stroke: C.faint });
  addLabel(slide, ctx, "Reward predictor", 682, 441, 200, 24, { size: 15, bold: true });
  addLabel(slide, ctx, "Linear → ReLU → Linear\nprediction target = y_score", 682, 478, 220, 36, {
    size: 12,
    color: C.muted,
  });

  addBox(slide, ctx, { x: 965, y: 235, w: 230, h: 128, fill: "#FFF8E8", stroke: "#E2C97E" });
  addLabel(slide, ctx, "Training signal", 986, 258, 180, 24, { size: 16, bold: true, color: C.gold });
  addLabel(slide, ctx, "Use only:\n(phi_x, phi_a_clinician) → y_score\n\nDo not use y_agent_score.", 986, 294, 180, 58, {
    size: 12,
    color: C.muted,
  });

  addBox(slide, ctx, { x: 965, y: 410, w: 230, h: 150, fill: "#F0F7F7", stroke: "#9AC5C8" });
  addLabel(slide, ctx, "Apply after training", 986, 433, 180, 24, { size: 16, bold: true, color: C.teal });
  addLabel(
    slide,
    ctx,
    "z_a_agent = action_projector(phi_a_agent)\n\nThen run DM / MIPS / OffCEM using the learned or concat-PCA features.",
    986,
    470,
    180,
    72,
    { size: 12, color: C.muted }
  );

  addLabel(
    slide,
    ctx,
    "In concat-PCA mode, the learned outputs are appended to PCA(BGE): final embedding = [PCA(BGE) ; learned_projection(BGE)].",
    88,
    645,
    1050,
    28,
    { size: 16, bold: true, color: C.ink, align: "center" }
  );
  addFooter(slide, ctx, "Implementation: src/cce_data/estimators/learned_embedding.py");
  return slide;
}
