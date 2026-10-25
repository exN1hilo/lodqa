// WebGL view of the LOD scene for the Streamlit app (st.components.v2).
//
// mode "orbit": a view of one precomputed frame. The game camera is drawn as a gizmo on its
//   path and the LOD choice of every method comes from Python. A guided tour flies to each
//   model and switches between the plans, with a pointer that shows what it clicks.
// mode "drive": the explorer. The viewer camera is the game camera, and the LOD choice of
//   DP and Greedy is recomputed in the browser (solver.js) whenever the camera moves.
//   QAOA results exist only for the precomputed path and for views run from the app.
//   With data.fullPage it covers the whole page under the app's top bar, and every text of
//   the page is one of its panels.

import * as THREE from "../vendor/three.module.min.js";
import { OrbitControls } from "../vendor/OrbitControls.js";

const QUERY = new URL(import.meta.url).search;
const solverReady = import(new URL(`./solver.js${QUERY}`, import.meta.url).href);
const CSS_URL = new URL(`./viewer.css${QUERY}`, import.meta.url).href;

const METHOD_ORDER = ["DP", "Greedy", "QAOA", "QAOA XY"];
const METHOD_LABEL = { DP: "Optimal", Greedy: "Greedy", QAOA: "QAOA · X", "QAOA XY": "QAOA · XY" };
const METHOD_LONG = {
  DP: "Optimal (dynamic programming)",
  Greedy: "Greedy shortcut",
  QAOA: "QAOA, X mixer",
  "QAOA XY": "QAOA, XY mixer",
};
const QUANTUM = new Set(["QAOA", "QAOA XY"]);
// How long one level of detail dissolves into the next.
const FADE_MS = 450;
// Length of each kind of tour step, and when the pointer clicks within a step.
const TOUR_MS = { overview: 4500, click: 2000, method: 2200, hold: 2400, summary: 7000 };
const CLICK_AT = { click: 700, method: 550 };
// The tour looks at the scene from closer than the default view.
const TOUR_DISTANCE = 1.5;

const THEMES = {
  light: {
    skyTop: "#d9e3ef",
    skyBottom: "#f5f4f0",
    fog: "#efeeea",
    ground: "#e9e7e1",
    grid: "#cfccc4",
    ink: "#0b0b0b",
    ink2: "#52514e",
    clay: "#d8d1c6",
    gray: "#c4c2bc",
    wire: "#0f1726",
    path: "#52514e",
    camera: "#0b0b0b",
  },
  dark: {
    skyTop: "#0c1320",
    skyBottom: "#1d2026",
    fog: "#1a1d22",
    ground: "#202328",
    grid: "#363a42",
    ink: "#f2f2f0",
    ink2: "#c3c1bb",
    clay: "#cbc3b7",
    gray: "#5d5b57",
    wire: "#e8eef8",
    path: "#c3c1bb",
    camera: "#f2f2f0",
  },
};

const ICONS = {
  play: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>',
  start: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M3 2.5h2v11H3zM14 2.5v11L6 8z"/></svg>',
  reset: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 8a5 5 0 1 0 1.5-3.6M3 2.5v2.5h2.5"/></svg>',
  label: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2.5 3.5h11v7h-6l-3 3v-3h-2z"/></svg>',
  target: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="5"/><circle cx="8" cy="8" r="1.4" fill="currentColor"/><path d="M8 1v2.2M8 12.8V15M1 8h2.2M12.8 8H15"/></svg>',
  palette: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 1.5a6.5 6.5 0 1 0 0 13c1 0 1.4-.6 1.4-1.3 0-1.2-1-1.3-1-2.3 0-.8.6-1.4 1.5-1.4h1.6a3 3 0 0 0 3-3C14.5 3.9 11.6 1.5 8 1.5z"/><circle cx="4.8" cy="7.2" r=".9" fill="currentColor"/><circle cx="7.2" cy="4.6" r=".9" fill="currentColor"/><circle cx="10.5" cy="5.2" r=".9" fill="currentColor"/></svg>',
  close: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
  pointer: '<svg viewBox="0 0 24 24"><path d="M4 2v17l4.6-4.3 3 6.8 2.8-1.2-3-6.7H18z" fill="#fcfcfb" stroke="#0b0b0b" stroke-width="1.4" stroke-linejoin="round"/></svg>',
};

const geometryCache = new Map();

function loadGeometry(url) {
  if (!geometryCache.has(url)) {
    const promise = fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
        return response.arrayBuffer();
      })
      .then((buffer) => {
        const header = new DataView(buffer, 0, 16);
        const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
        if (magic !== "LODM" || header.getUint32(4, true) !== 1) throw new Error(`${url}: not a LODM mesh`);
        const nVertices = header.getUint32(8, true);
        const nFaces = header.getUint32(12, true);
        let offset = 16;
        const positions = new Float32Array(buffer, offset, nVertices * 3);
        offset += nVertices * 12;
        const normals = new Float32Array(buffer, offset, nVertices * 3);
        offset += nVertices * 12;
        const indices = new Uint32Array(buffer, offset, nFaces * 3);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeBoundingSphere();
        geometry.userData.triangles = nFaces;
        return geometry;
      });
    promise.catch(() => geometryCache.delete(url));
    geometryCache.set(url, promise);
  }
  return geometryCache.get(url);
}

const fmtInt = new Intl.NumberFormat("en-US");
const fmtK = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : `${n}`);
const pct = (x) => `${Math.round(x * 100)}%`;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const vec = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const sameChoice = (a, b) => !!a && !!b && a.length === b.length && a.every((x, i) => x === b[i]);

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function button(label, { icon, title, className = "lv-btn", pressed } = {}) {
  const node = el("button", className, `${icon ? ICONS[icon] : ""}${label ? `<span>${label}</span>` : ""}`);
  node.type = "button";
  if (title) {
    node.title = title;
    node.setAttribute("aria-label", title);
  }
  if (pressed !== undefined) node.setAttribute("aria-pressed", String(pressed));
  return node;
}

function gradientTexture(top, bottom) {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 256);
  gradient.addColorStop(0, top);
  gradient.addColorStop(1, bottom);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

class LodViewer {
  constructor(root, data, solver) {
    this.root = root;
    this.solver = solver;
    this.data = data;
    this.mode = data.mode;
    this.theme = THEMES[data.theme] ?? THEMES.light;
    this.trigger = () => {};
    this.setState = () => {};
    this.tour = null;
    this.reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
    this.disposed = false;
    this.needsRender = true;
    this.state = {
      method: data.method ?? "DP",
      budget: data.budget?.value,
      frame: data.frame ?? 0,
      pathT: data.frame ?? 0,
      playing: false,
      lodColors: true,
      labels: true,
      focused: null,
      pending: null,
      onPath: true,
    };
    this.applied = { budget: data.budget?.value, method: data.method, frame: data.frame };
    this.current = null;
    this.lastSolveKey = "";
    this.lastHud = 0;
    this.keys = new Set();
    this.tween = null;
    this.buildDom();
    this.buildThree();
    this.buildScene().catch((error) => this.fail(error));
    this.loop = this.loop.bind(this);
    this.clock = performance.now();
    this.frameId = requestAnimationFrame(this.loop);
  }

  // ---------------------------------------------------------------- DOM
  buildDom() {
    if (!this.root.querySelector("link[data-lv]")) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = CSS_URL;
      link.dataset.lv = "1";
      this.root.appendChild(link);
    }
    const wrap = el("div", "lv");
    wrap.dataset.theme = this.data.theme === "dark" ? "dark" : "light";
    wrap.dataset.mode = this.mode;
    wrap.tabIndex = 0;
    wrap.style.setProperty("--lv-height", `${this.data.height ?? 560}px`);
    wrap.setAttribute("role", "application");
    wrap.setAttribute(
      "aria-label",
      this.mode === "drive"
        ? "3D explorer. Drag to look around, W A S D to move, click a model to fly to it."
        : "3D view of the scene. Drag to orbit, scroll to zoom, click a model to zoom in.",
    );
    this.root.appendChild(wrap);
    this.wrap = wrap;

    this.glHost = el("div", "lv-gl");
    wrap.appendChild(this.glHost);
    this.labelsHost = el("div", "lv-labels");
    wrap.appendChild(this.labelsHost);
    this.status = el("div", "lv-status", '<div><div class="lv-spinner"></div>Loading 3D models…</div>');
    wrap.appendChild(this.status);
    // Top centre: the explorer's intro and the toasts; right: a column of panels.
    this.top = el("div", "lv-top");
    this.toasts = el("div", "lv-toasts");
    this.top.appendChild(this.toasts);
    wrap.appendChild(this.top);
    this.side = el("div", "lv-side");
    wrap.appendChild(this.side);

    this.buildToolbar();
    if (this.mode === "drive") this.buildDriveHud();
    else this.buildOrbitHud();
    this.buildFocusCard();
    if (this.mode === "drive") this.buildRuns();
    if (this.mode === "orbit") this.buildTour();
    if (this.data.fullPage) this.buildFullPage();
  }

  buildFullPage() {
    this.wrap.dataset.full = "true";
    this.onWindowResize = () => this.placeFull();
    window.addEventListener("resize", this.onWindowResize);
    const bar = document.querySelector('header[data-testid="stHeader"]');
    if (bar) {
      this.barObserver = new ResizeObserver(this.onWindowResize);
      this.barObserver.observe(bar);
    }
    this.placeFull();
  }

  // The view is fixed under the app's top bar, wherever the bar ends.
  placeFull() {
    const bar = document.querySelector('header[data-testid="stHeader"]');
    const top = bar ? Math.max(0, Math.round(bar.getBoundingClientRect().bottom)) : 0;
    this.wrap.style.setProperty("--lv-top", `${top}px`);
  }

  buildToolbar() {
    const bar = el("div", "lv-panel lv-toolbar");
    this.btnColors = button("", {
      icon: "palette",
      title: "Color models by level of detail (C)",
      className: "lv-btn lv-icon",
      pressed: true,
    });
    this.btnLabels = button("", { icon: "label", title: "Show labels (L)", className: "lv-btn lv-icon", pressed: true });
    this.btnReset = button("", { icon: "reset", title: "Reset the view (R)", className: "lv-btn lv-icon" });
    bar.append(this.btnColors, this.btnLabels, this.btnReset);
    this.wrap.appendChild(bar);

    this.btnColors.onclick = () => this.setLodColors(!this.state.lodColors);
    this.btnLabels.onclick = () => this.setLabels(!this.state.labels);
    this.btnReset.onclick = () => {
      this.pauseTour();
      this.resetView();
    };
  }

  buildOrbitHud() {
    const legend = el("div", "lv-panel lv-legend");
    const last = this.data.lodColors.length - 1;
    const keys = this.data.lodColors
      .map((color, j) => `<span class="lv-key"><span class="lv-dot" style="background:${color}"></span>LOD${j}${j === 0 ? " most detail" : j === last ? " least" : ""}</span>`)
      .join("");
    legend.innerHTML = `
      <div class="lv-seg" role="group" aria-label="Plan shown"></div>
      <div class="lv-keys">${keys}</div>
      <div class="lv-small lv-muted lv-frame-text"></div>
      <div class="lv-small lv-muted lv-orbit-note lv-hidden"></div>`;
    this.wrap.appendChild(legend);
    this.legend = legend;

    const seg = legend.querySelector(".lv-seg");
    this.methodButtons = {};
    METHOD_ORDER.forEach((method, k) => {
      const b = button(METHOD_LABEL[method], { className: "", title: `Show the plan of ${METHOD_LONG[method]} (${k + 1})`, pressed: false });
      b.onclick = () => this.pickMethod(method);
      seg.appendChild(b);
      this.methodButtons[method] = b;
    });
  }

  buildTour() {
    const panel = el("div", "lv-panel lv-tour");
    this.btnTour = button("Play", { icon: "play", title: "Play (Space)", className: "lv-btn lv-primary lv-tour-play" });
    this.btnTour.disabled = true;
    this.btnTour.onclick = () => this.toggleTour();
    const body = el("div", "lv-tour-body", '<div class="lv-tour-text" aria-live="polite"></div><div class="lv-tour-track" aria-hidden="true"><span></span></div>');
    panel.append(this.btnTour, body);
    this.wrap.appendChild(panel);
    this.tourText = body.querySelector(".lv-tour-text");
    this.tourBar = body.querySelector(".lv-tour-track span");
    this.cursor = el("div", "lv-cursor lv-cursor-off", ICONS.pointer);
    this.wrap.appendChild(this.cursor);
    this.renderTour();
  }

  buildDriveHud() {
    const main = el("div", "lv-panel lv-main");
    main.innerHTML = `
      <div class="lv-heading"><strong>You are the game camera</strong><span class="lv-small lv-muted lv-where"></span></div>
      <label class="lv-row lv-small lv-scene-row lv-hidden"><span class="lv-muted">Scene</span><select class="lv-select" aria-label="Scene"></select></label>
      <div class="lv-seg" role="group" aria-label="Method"></div>
      <div>
        <div class="lv-row"><span>Triangle budget</span><b class="lv-budget-value"></b></div>
        <input class="lv-range lv-budget" type="range" aria-label="Triangle budget">
      </div>
      <div>
        <div class="lv-bar lv-used-bar" aria-hidden="true"></div>
        <div class="lv-row lv-small lv-muted" style="margin-top:4px"><span class="lv-used"></span><span class="lv-quality-text"></span></div>
      </div>
      <div class="lv-quality"></div>
      <div class="lv-quantum lv-hidden"></div>`;
    this.wrap.appendChild(main);
    this.main = main;

    // A new scene is a new problem: Python reloads the page data for it.
    const scenes = this.data.scenes ?? [];
    if (scenes.length) {
      const select = main.querySelector(".lv-select");
      select.innerHTML = scenes.map((name) => `<option>${name}</option>`).join("");
      select.value = this.data.sceneKey;
      select.onchange = () => {
        this.setPlaying(false);
        select.disabled = true;
        this.trigger("scene", select.value);
      };
      main.querySelector(".lv-scene-row").classList.remove("lv-hidden");
    }
    if (this.data.fullPage) this.buildIntro();

    const seg = main.querySelector(".lv-seg");
    this.methodButtons = {};
    for (const method of METHOD_ORDER) {
      const b = button(METHOD_LABEL[method], { className: "", title: METHOD_LONG[method], pressed: false });
      b.onclick = () => this.setMethod(method);
      seg.appendChild(b);
      this.methodButtons[method] = b;
    }
    const budget = main.querySelector(".lv-budget");
    budget.min = this.data.budget.min;
    budget.max = this.data.budget.max;
    budget.step = this.data.budget.step;
    budget.value = this.state.budget;
    budget.oninput = () => {
      this.state.budget = Number(budget.value);
      this.needsRender = true;
    };
    this.budgetInput = budget;

    this.quality = main.querySelector(".lv-quality");
    this.scoreCells = {};
    for (const method of METHOD_ORDER) {
      const cell = el(
        "div",
        "lv-score",
        `<span class="lv-score-name">${METHOD_LABEL[method]}</span><span class="lv-score-value">–</span><span class="lv-score-note"></span>`,
      );
      cell.title = `Picture quality of ${METHOD_LONG[method]}, relative to the optimum`;
      cell.onclick = () => this.setMethod(method);
      this.quality.appendChild(cell);
      this.scoreCells[method] = cell;
    }
    this.quantumBox = main.querySelector(".lv-quantum");

    const objects = el("div", "lv-panel lv-objects");
    objects.innerHTML = `<table><thead><tr><th>Model</th><th>Detail</th><th class="lv-num">Triangles</th><th class="lv-num">Distance</th><th></th></tr></thead><tbody></tbody></table>`;
    this.wrap.appendChild(objects);
    this.objectRows = objects.querySelector("tbody");

    const minimap = el("div", "lv-panel lv-minimap");
    this.minimap = document.createElement("canvas");
    minimap.appendChild(this.minimap);
    minimap.appendChild(el("div", "lv-caption", "Map from above · ▲ is you"));
    this.side.appendChild(minimap);

    const player = el("div", "lv-panel lv-player");
    this.btnStart = button("", { icon: "start", title: "Back to the start of the path", className: "lv-btn lv-icon" });
    this.btnPlay = button("", { icon: "play", title: "Play the camera path (Space)", className: "lv-btn lv-icon" });
    this.pathInput = el("input", "lv-range");
    this.pathInput.type = "range";
    this.pathInput.min = 0;
    this.pathInput.max = this.data.path.length - 1;
    this.pathInput.step = 0.01;
    this.pathInput.value = this.state.pathT;
    this.pathInput.setAttribute("aria-label", "Position on the camera path");
    this.frameText = el("span", "lv-frame");
    player.append(this.btnStart, this.btnPlay, this.pathInput, this.frameText);
    this.wrap.appendChild(player);
    this.btnStart.onclick = () => {
      this.setPlaying(false);
      this.goToPath(0, true);
    };
    this.btnPlay.onclick = () => this.setPlaying(!this.state.playing);
    this.pathInput.oninput = () => {
      this.setPlaying(false);
      this.goToPath(Number(this.pathInput.value), false);
    };
  }

  buildFocusCard() {
    this.focusCard = el("div", "lv-panel lv-focus lv-hidden");
    this.side.appendChild(this.focusCard);
  }

  buildIntro() {
    const key = "lodq-explorer-intro";
    try {
      if (localStorage.getItem(key) === "hidden") return;
    } catch {}
    this.intro = el(
      "div",
      "lv-panel lv-intro",
      `<div><strong>Fly through the scene.</strong> Drag to look around, scroll or press <kbd>W</kbd>/<kbd>S</kbd> to move, and click any model to fly to it and see its triangles. Every time the camera moves, your browser solves the budget problem again, exactly (DP) and with the greedy shortcut, and swaps each model to the level of detail it gets.</div>`,
    );
    const close = button("", { icon: "close", title: "Hide this note", className: "lv-btn lv-icon" });
    close.onclick = () => {
      this.intro.remove();
      try {
        localStorage.setItem(key, "hidden");
      } catch {}
    };
    this.intro.appendChild(close);
    this.top.prepend(this.intro);
  }

  // The views the user ran QAOA on, in this scene.
  buildRuns() {
    this.runsPanel = el("div", "lv-panel lv-runs lv-hidden");
    this.side.appendChild(this.runsPanel);
    this.renderRuns();
  }

  renderRuns() {
    const runs = this.data.runs ?? [];
    this.runsPanel.classList.toggle("lv-hidden", !runs.length);
    if (!runs.length) return;
    const head = [
      ["Camera", ""],
      ["Budget", ""],
      ["Greedy", "Quality of the greedy plan, relative to the optimum"],
      ["X", "Quality of the top answer of QAOA with the X mixer"],
      ["XY", "Quality of the top answer of QAOA with the XY mixer"],
      ["XY opt.", "Shots of the XY circuit that are the optimal plan"],
      ["s", "Seconds the run took"],
    ]
      .map(([name, title], k) => `<th${k ? ' class="lv-num"' : ""}${title ? ` title="${title}"` : ""}>${name}</th>`)
      .join("");
    const rows = runs
      .map(
        (run) =>
          `<tr><td>${run.camera}</td>${[run.budget, run.greedy, run.x, run.xy, run.xyOptimum, run.seconds].map((value) => `<td class="lv-num">${value}</td>`).join("")}</tr>`,
      )
      .join("");
    this.runsPanel.innerHTML = `<strong>Views you ran QAOA on</strong><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  // -------------------------------------------------------------- three
  buildThree() {
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.glHost.appendChild(renderer.domElement);
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(this.data.scene.fov ?? 60, 1, 0.05, 800);
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener("change", () => (this.needsRender = true));
    this.controls.addEventListener("start", () => {
      this.tween = null;
      if (this.mode === "drive") this.setPlaying(false);
      this.pauseTour();
    });

    this.raycaster = new THREE.Raycaster();
    this.pointer = { x: 0, y: 0, down: null };
    const canvas = renderer.domElement;
    canvas.addEventListener("pointerdown", (event) => {
      this.pointer.down = { x: event.clientX, y: event.clientY };
      this.wrap.focus({ preventScroll: true });
    });
    canvas.addEventListener("pointerup", (event) => {
      const down = this.pointer.down;
      this.pointer.down = null;
      if (!down || Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) return;
      const hit = this.pick(event);
      if (hit !== null) {
        this.pauseTour();
        this.focus(hit);
      }
    });
    canvas.addEventListener("pointermove", (event) => {
      if (this.pointer.down) return;
      const hit = this.pick(event);
      canvas.style.cursor = hit === null ? "" : "pointer";
    });

    this.onKeyDown = (event) => this.keyDown(event);
    this.onKeyUp = (event) => this.keys.delete(event.code);
    this.wrap.addEventListener("keydown", this.onKeyDown);
    this.wrap.addEventListener("keyup", this.onKeyUp);
    this.wrap.addEventListener("blur", () => this.keys.clear());

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.wrap);
    this.visible = true;
    this.intersection = new IntersectionObserver((entries) => {
      this.visible = entries.some((entry) => entry.isIntersecting);
      if (this.visible) this.needsRender = true;
    });
    this.intersection.observe(this.wrap);
    this.resize();
  }

  async buildScene() {
    const scene = this.data.scene;
    const theme = this.theme;
    const objects = scene.objects;

    // Bounds of the models and the camera path.
    const box = new THREE.Box3();
    for (const obj of objects) {
      box.expandByPoint(vec(obj.position).addScalar(obj.radius));
      box.expandByPoint(vec(obj.position).addScalar(-obj.radius));
    }
    const modelBox = box.clone();
    for (const p of this.data.path) box.expandByPoint(vec(p));
    this.center = modelBox.getCenter(new THREE.Vector3());
    this.radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1);
    this.groundY = modelBox.min.y - 0.35 * Math.max(...objects.map((o) => o.radius));
    const R = this.radius;

    this.scene.background = gradientTexture(theme.skyTop, theme.skyBottom);
    this.scene.fog = new THREE.Fog(theme.fog, R * 2.2, R * 7);

    this.scene.add(new THREE.HemisphereLight(0xffffff, new THREE.Color(theme.ground), 1.15));
    const sun = new THREE.DirectionalLight(0xfff3e2, 2.4);
    sun.position.copy(this.center).add(new THREE.Vector3(-0.6 * R, 1.6 * R, -0.9 * R));
    sun.target.position.copy(this.center);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const extent = R * 1.1;
    Object.assign(sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near: 0.1, far: R * 5 });
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun, sun.target);
    const fill = new THREE.DirectionalLight(0xdbe7ff, 0.7);
    fill.position.copy(this.center).add(new THREE.Vector3(0.9 * R, 0.4 * R, 0.8 * R));
    this.scene.add(fill);

    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(R * 6, 96),
      new THREE.MeshStandardMaterial({ color: theme.ground, roughness: 1, metalness: 0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(this.center.x, this.groundY, this.center.z);
    ground.receiveShadow = true;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(R * 8, 64, theme.grid, theme.grid);
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    grid.material.depthWrite = false;
    grid.position.set(this.center.x, this.groundY + 0.01, this.center.z);
    this.scene.add(grid);

    this.buildPath();

    // Models: one group per object with every LOD mesh, only the chosen one visible.
    const geometries = await Promise.all(objects.map((obj) => Promise.all(obj.meshes.map(loadGeometry))));
    if (this.disposed) return;
    this.wireMaterial = new THREE.MeshBasicMaterial({
      color: theme.wire,
      wireframe: true,
      transparent: true,
      opacity: this.data.theme === "dark" ? 0.28 : 0.22,
      depthWrite: false,
    });
    this.models = objects.map((obj, i) => {
      const group = new THREE.Group();
      group.position.copy(vec(obj.position));
      group.scale.setScalar(obj.radius);
      // Hashed alpha dissolves one LOD into the next without sorting transparent meshes;
      // at opacity 1 the material is plain opaque.
      const material = new THREE.MeshStandardMaterial({
        color: theme.clay,
        roughness: 0.58,
        metalness: 0.05,
        alphaHash: true,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      const meshes = geometries[i].map((geometry) => {
        const mesh = new THREE.Mesh(geometry, material.clone());
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.visible = false;
        mesh.userData.index = i;
        const wire = new THREE.Mesh(geometry, this.wireMaterial);
        wire.visible = false;
        wire.raycast = () => {};
        mesh.add(wire);
        mesh.userData.wire = wire;
        group.add(mesh);
        return mesh;
      });
      this.scene.add(group);

      // A dashed stem to the ground and a ring under it make the height readable.
      const bottom = obj.position[1] - obj.radius;
      const stem = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(obj.position[0], bottom, obj.position[2]),
          new THREE.Vector3(obj.position[0], this.groundY, obj.position[2]),
        ]),
        new THREE.LineDashedMaterial({ color: theme.ink2, dashSize: 0.25, gapSize: 0.2, transparent: true, opacity: 0.45 }),
      );
      stem.computeLineDistances();
      this.scene.add(stem);
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(obj.radius * 0.82, obj.radius * 0.92, 48),
        new THREE.MeshBasicMaterial({ color: theme.ink2, transparent: true, opacity: 0.22, depthWrite: false }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(obj.position[0], this.groundY + 0.02, obj.position[2]);
      this.scene.add(ring);

      const label = el("div", "lv-label");
      label.onclick = () => {
        this.pauseTour();
        this.focus(i);
      };
      this.labelsHost.appendChild(label);
      return { group, meshes, label, level: null, shown: null, fade: null, ring };
    });

    this.status.classList.add("lv-hidden");
    this.resetView(true);
    this.refresh(true);
    if (this.mode === "orbit") this.renderTour();
  }

  buildPath() {
    const theme = this.theme;
    const points = this.data.path.map(vec);
    if (points.length > 1) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineDashedMaterial({ color: theme.path, dashSize: 0.35, gapSize: 0.25, transparent: true, opacity: 0.7 }),
      );
      line.computeLineDistances();
      this.scene.add(line);
    }
    // Frame markers only from outside: in the explorer the next marker sits right in front
    // of the lens and would cover the view.
    if (this.mode === "orbit") {
      const dots = new THREE.InstancedMesh(
        new THREE.SphereGeometry(0.06, 10, 8),
        new THREE.MeshBasicMaterial({ color: theme.path }),
        points.length,
      );
      const matrix = new THREE.Matrix4();
      points.forEach((p, k) => dots.setMatrixAt(k, matrix.makeTranslation(p.x, p.y, p.z)));
      this.scene.add(dots);
    }

    // The game camera: a small body and its viewing pyramid (orbit mode only).
    this.gizmo = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.ConeGeometry(0.22, 0.45, 4),
      new THREE.MeshStandardMaterial({ color: theme.camera, roughness: 0.4 }),
    );
    body.rotation.x = Math.PI / 2;
    body.rotation.y = Math.PI / 4;
    body.position.z = 0.2;
    this.gizmo.add(body);
    const half = Math.tan(THREE.MathUtils.degToRad(this.data.scene.fov ?? 60) / 2);
    const depth = 2.2;
    const aspect = 16 / 9;
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([sx, sy]) => new THREE.Vector3(sx * half * aspect * depth, sy * half * depth, depth));
    const segments = [];
    for (const c of corners) segments.push(new THREE.Vector3(), c);
    for (let k = 0; k < 4; k += 1) segments.push(corners[k], corners[(k + 1) % 4]);
    this.gizmo.add(
      new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(segments),
        new THREE.LineBasicMaterial({ color: theme.camera, transparent: true, opacity: 0.75 }),
      ),
    );
    this.gizmo.visible = this.mode === "orbit";
    this.scene.add(this.gizmo);
  }

  // ------------------------------------------------------------ updates
  update(data) {
    const sceneChanged = data.scene.name !== this.data.scene.name || data.mode !== this.mode;
    if (sceneChanged) return false;
    const prev = this.data;
    this.data = data;
    // The tour was planned for the old frame; a new frame or budget ends it.
    const problemChanged =
      data.frame !== prev.frame || data.budget.value !== prev.budget.value || JSON.stringify(data.choices) !== JSON.stringify(prev.choices);
    if (this.tour && problemChanged) this.stopTour();
    if (data.method !== this.applied.method) {
      this.applied.method = data.method;
      this.state.method = data.method;
    }
    if (this.mode === "drive") {
      if (data.budget.value !== this.applied.budget) {
        this.applied.budget = data.budget.value;
        this.state.budget = data.budget.value;
        this.budgetInput.value = data.budget.value;
      }
      this.budgetInput.min = data.budget.min;
      this.budgetInput.max = data.budget.max;
      if (data.frame !== this.applied.frame) {
        this.applied.frame = data.frame;
        this.setPlaying(false);
        this.goToPath(data.frame, true);
      }
      if (this.state.pending && this.quantumAt(this.state.pending.camera, this.state.pending.budget)) {
        this.state.pending = null;
      }
      this.renderRuns();
    } else if (data.frame !== prev.frame) {
      this.moveGizmo(true);
    }
    this.lastSolveKey = "";
    this.refresh(true);
    if (this.mode === "orbit") this.renderTour();
    return true;
  }

  setMethod(method) {
    this.state.method = method;
    this.lastSolveKey = "";
    this.refresh(true);
  }

  // A method picked by the user in orbit mode; Python draws the map of the same plan.
  pickMethod(method) {
    this.pauseTour();
    if (this.tour) this.tour.returnMethod = method;
    this.applied.method = method;
    this.setMethod(method);
    this.setState("method", method);
  }

  setLodColors(on) {
    this.state.lodColors = on;
    this.btnColors.setAttribute("aria-pressed", String(on));
    this.models?.forEach((model, i) => this.paint(i));
    this.needsRender = true;
  }

  setLabels(on) {
    this.state.labels = on;
    this.btnLabels.setAttribute("aria-pressed", String(on));
    this.labelsHost.classList.toggle("lv-hidden", !on);
  }

  setPlaying(on) {
    if (this.mode !== "drive") return;
    this.state.playing = on;
    this.btnPlay.innerHTML = on ? ICONS.pause : ICONS.play;
    this.btnPlay.title = on ? "Pause (Space)" : "Play the camera path (Space)";
    if (on) {
      this.tween = null;
      if (this.state.pathT >= this.data.path.length - 1) this.state.pathT = 0;
      this.goToPath(this.state.pathT, false);
    }
  }

  pathPosition(t) {
    const path = this.data.path;
    const k = Math.min(Math.max(Math.floor(t), 0), path.length - 1);
    const next = Math.min(k + 1, path.length - 1);
    return vec(path[k]).lerp(vec(path[next]), t - k);
  }

  goToPath(t, animate) {
    const last = this.data.path.length - 1;
    this.state.pathT = Math.min(Math.max(t, 0), last);
    this.state.onPath = true;
    const position = this.pathPosition(this.state.pathT);
    const target = vec(this.data.scene.target);
    if (animate) this.flyTo(position, target, 700);
    else {
      this.tween = null;
      this.camera.position.copy(position);
      this.controls.target.copy(target);
      this.controls.update();
    }
    if (this.pathInput) this.pathInput.value = this.state.pathT;
    this.needsRender = true;
  }

  moveGizmo(animate) {
    const frame = this.data.frame ?? 0;
    const position = vec(this.data.path[frame] ?? this.data.path[0]);
    const target = vec(this.data.scene.target);
    if (animate && this.gizmo.visible) {
      this.gizmoTween = { from: this.gizmo.position.clone(), to: position, start: performance.now(), duration: 350, target };
    } else {
      this.gizmo.position.copy(position);
      this.gizmo.lookAt(target);
    }
    this.needsRender = true;
  }

  resetView(initial = false, distance = 2.1) {
    if (!this.center) return;
    this.clearFocus();
    if (this.mode === "drive") {
      this.goToPath(initial ? this.state.pathT : Math.round(this.state.pathT), !initial);
      this.controls.minDistance = 0.3;
      this.controls.maxDistance = this.radius * 3;
      return;
    }
    // Frame the models, seen from the side of the game camera and a little above it; the
    // camera gizmo comes into view as the camera gets closer along its path. `distance` is
    // in units of the half diagonal of the models' bounding box.
    const box = new THREE.Box3();
    for (const obj of this.data.scene.objects) {
      box.expandByPoint(vec(obj.position).addScalar(obj.radius));
      box.expandByPoint(vec(obj.position).addScalar(-obj.radius));
    }
    const center = box.getCenter(new THREE.Vector3());
    const R = box.getSize(new THREE.Vector3()).length() / 2;
    const eye = center.clone().add(new THREE.Vector3(-0.5, 0.55, -1).normalize().multiplyScalar(R * distance));
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = this.radius * 5;
    if (initial) {
      this.camera.position.copy(eye);
      this.controls.target.copy(center);
      this.controls.update();
      this.moveGizmo(false);
    } else {
      this.flyTo(eye, center, 700);
    }
  }

  flyTo(position, target, duration = 900) {
    this.tween = {
      fromPos: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      toPos: position.clone(),
      toTarget: target.clone(),
      start: performance.now(),
      duration,
    };
    this.needsRender = true;
  }

  focus(i) {
    const obj = this.data.scene.objects[i];
    const center = vec(obj.position);
    const direction = this.camera.position.clone().sub(center);
    if (direction.lengthSq() < 1e-9) direction.set(0, 0.3, -1);
    direction.normalize();
    direction.y = Math.max(direction.y, 0.18);
    direction.normalize();
    const distance = obj.radius * (this.mode === "drive" ? 2.6 : 3.2);
    this.setPlaying(false);
    this.state.onPath = false;
    this.state.focused = i;
    this.flyTo(center.clone().add(direction.multiplyScalar(distance)), center, 1000);
    this.applyWire();
    this.renderFocusCard();
    this.renderObjects();
  }

  clearFocus() {
    if (this.state.focused === null) return;
    this.state.focused = null;
    this.focusCard.classList.add("lv-hidden");
    this.applyWire();
    this.renderObjects();
  }

  applyWire() {
    this.models?.forEach((model, i) => {
      const on = this.state.focused === i;
      model.meshes.forEach((mesh, j) => (mesh.userData.wire.visible = on && j === model.shown));
    });
    this.needsRender = true;
  }

  // Cross-fade from the mesh on screen to LOD `to`; a fade still running is finished first.
  startFade(model, to) {
    if (model.fade) this.endFade(model);
    const from = model.shown;
    model.fade = { from, to, start: performance.now() };
    model.shown = to;
    model.meshes[from].visible = true;
    model.meshes[to].visible = true;
    model.meshes[to].material.opacity = 0;
  }

  endFade(model) {
    model.fade = null;
    model.meshes.forEach((mesh, j) => {
      mesh.visible = j === model.shown;
      mesh.material.opacity = 1;
    });
  }

  stepFades(now) {
    for (const model of this.models) {
      if (!model.fade) continue;
      const t = Math.min((now - model.fade.start) / FADE_MS, 1);
      const e = easeInOut(t);
      model.meshes[model.fade.from].material.opacity = 1 - e;
      model.meshes[model.fade.to].material.opacity = e;
      if (t >= 1) this.endFade(model);
      this.needsRender = true;
    }
  }

  keyDown(event) {
    if (event.target !== this.wrap) return;
    const code = event.code;
    const movement = ["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "ShiftLeft", "ShiftRight"];
    if (this.mode === "drive" && movement.includes(code)) {
      this.keys.add(code);
      if (code !== "ShiftLeft" && code !== "ShiftRight") {
        this.setPlaying(false);
        this.tween = null;
      }
      event.preventDefault();
      return;
    }
    const drive = this.mode === "drive";
    const method = (k) => () => (drive ? this.setMethod(METHOD_ORDER[k]) : this.pickMethod(METHOD_ORDER[k]));
    const actions = {
      Space: () => (drive ? this.setPlaying(!this.state.playing) : this.toggleTour()),
      KeyC: () => this.setLodColors(!this.state.lodColors),
      KeyL: () => this.setLabels(!this.state.labels),
      KeyR: () => (this.pauseTour(), this.resetView()),
      Escape: () => (this.pauseTour(), this.clearFocus()),
      Digit1: method(0),
      Digit2: method(1),
      Digit3: method(2),
      Digit4: method(3),
    };
    if (actions[code]) {
      actions[code]();
      event.preventDefault();
    }
  }

  pick(event) {
    if (!this.models) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const visible = this.models.flatMap((model) => model.meshes.filter((mesh) => mesh.visible));
    const hit = this.raycaster.intersectObjects(visible, false)[0];
    return hit ? hit.object.userData.index : null;
  }

  resize() {
    const width = this.wrap.clientWidth;
    const height = this.wrap.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.needsRender = true;
  }

  // --------------------------------------------------------------- tour
  // The guided tour is a list of timed steps, each with actions at fixed offsets, so a
  // paused tour restarts its current step. It switches methods in the viewer only.
  tourSteps() {
    const objects = this.data.scene.objects;
    const choices = this.data.choices ?? {};
    const methods = METHOD_ORDER.filter((m) => choices[m]);
    const camera = this.cameraArray();
    const distances = objects.map((obj) => this.solver.distance(obj.position, camera));
    const varies = objects.map((obj, i) => new Set(methods.map((m) => choices[m][i])).size > 1);
    // Models whose detail depends on the method come first, nearest first.
    const order = objects.map((obj, i) => i).sort((a, b) => varies[b] - varies[a] || distances[a] - distances[b]);
    const steps = [{ kind: "overview" }];
    for (const i of order) {
      steps.push({ kind: "click", i });
      if (varies[i]) for (const method of methods) steps.push({ kind: "method", i, method });
      else steps.push({ kind: "hold", i });
    }
    steps.push({ kind: "summary" });
    return steps.map((step) => ({ ...step, ms: TOUR_MS[step.kind] }));
  }

  toggleTour() {
    if (this.mode !== "orbit" || !this.models) return;
    const state = this.tour?.state;
    if (state === "playing") this.pauseTour();
    else if (state === "paused") this.resumeTour();
    else this.startTour();
  }

  startTour() {
    const steps = this.tourSteps();
    this.tour = {
      steps,
      total: steps.reduce((sum, step) => sum + step.ms, 0),
      index: 0,
      state: "playing",
      returnMethod: this.state.method,
    };
    this.placeCursor(this.wrap.clientWidth / 2, this.wrap.clientHeight * 0.7);
    this.enterStep();
  }

  pauseTour() {
    if (this.tour?.state !== "playing") return;
    this.tour.state = "paused";
    this.hideCursor();
    this.renderTour();
  }

  resumeTour() {
    this.tour.state = "playing";
    this.enterStep();
  }

  stopTour() {
    if (!this.tour) return;
    if (this.tour.state !== "ended") this.setMethod(this.tour.returnMethod);
    this.tour = null;
    this.hideCursor();
    this.renderTour();
  }

  enterStep() {
    const tour = this.tour;
    const step = tour.steps[tour.index];
    tour.start = performance.now();
    tour.done = tour.steps.slice(0, tour.index).reduce((sum, s) => sum + s.ms, 0);
    // A model off screen is clicked from the overview: fly out first, then aim.
    tour.delay = step.kind === "click" && !this.onScreen(step.i) ? 800 : 0;
    tour.actions = this.stepActions(step, tour.delay);
    this.renderTour();
  }

  stepActions(step, delay) {
    const narrate = () => this.narrate(step);
    switch (step.kind) {
      case "overview":
        return [{ at: 0, run: () => (this.hideCursor(), this.resetView(false, TOUR_DISTANCE), narrate()) }];
      case "click":
        return [
          ...(delay ? [{ at: 0, run: () => this.resetView(false, TOUR_DISTANCE) }] : []),
          { at: delay, run: () => this.aimCursor(this.screenPoint(step.i)) },
          { at: delay + CLICK_AT.click, run: () => (this.focus(step.i), narrate()) },
        ];
      case "method":
        return [
          { at: 0, run: () => this.aimCursor(this.elementPoint(this.methodButtons[step.method])) },
          { at: CLICK_AT.method, run: () => (this.setMethod(step.method), narrate()) },
        ];
      case "hold":
        return [{ at: 0, run: narrate }];
      default:
        return [{ at: 0, run: () => (this.hideCursor(), this.setMethod(this.tour.returnMethod), this.resetView(false, TOUR_DISTANCE), narrate()) }];
    }
  }

  stepTour(now) {
    const tour = this.tour;
    if (tour?.state !== "playing") return;
    const step = tour.steps[tour.index];
    const elapsed = now - tour.start;
    for (const action of tour.actions) {
      if (!action.done && elapsed >= action.at) {
        action.done = true;
        action.run();
      }
    }
    if (elapsed >= step.ms + tour.delay) {
      if (tour.index + 1 < tour.steps.length) {
        tour.index += 1;
        this.enterStep();
      } else {
        tour.state = "ended";
        this.renderTour();
      }
      return;
    }
    this.tourBar.style.width = `${(100 * (tour.done + Math.min(elapsed, step.ms))) / tour.total}%`;
  }

  renderTour() {
    if (!this.btnTour) return;
    const tour = this.tour;
    const state = tour?.state ?? "idle";
    const playing = state === "playing";
    const label = playing ? "Pause" : "Play";
    this.btnTour.innerHTML = `${ICONS[playing ? "pause" : "play"]}<span>${label}</span>`;
    this.btnTour.title = `${label} (Space)`;
    this.btnTour.setAttribute("aria-label", this.btnTour.title);
    this.btnTour.disabled = !this.models;
    // Before the first play there is nothing to narrate, so only the button shows.
    this.wrap.dataset.tour = state;
    if (state === "idle") this.tourText.innerHTML = "";
    const done = state === "ended" ? tour.total : (tour?.done ?? 0);
    this.tourBar.style.width = tour ? `${(100 * done) / tour.total}%` : "0%";
  }

  narrate(step) {
    const { title, text } = this.tourCopy(step);
    this.tourText.innerHTML = `<strong>${title}</strong> <span>${text}</span>`;
  }

  tourCopy(step) {
    const objects = this.data.scene.objects;
    const obj = objects[step.i];
    const choices = this.data.choices ?? {};
    const solutions = this.current?.solutions ?? {};
    const quality = (m) => (solutions[m] && solutions.DP ? solutions[m].value / solutions.DP.value : null);
    const triangles = (level) => fmtInt.format(obj.triCounts[level]);
    switch (step.kind) {
      case "overview":
        return {
          title: "The scene",
          text: `${objects.length} models with ${objects[0].triCounts.length} levels of detail each. The game camera flies along the dashed path and can draw <b>${fmtInt.format(this.data.budget.value)}</b> triangles in this frame. Which models get the detail?`,
        };
      case "click": {
        const distance = this.solver.distance(obj.position, this.cameraArray());
        const meshes = obj.triCounts.map((count, j) => `LOD${j} ${fmtK(count)}`).join(" · ");
        return { title: obj.name, text: `${distance.toFixed(1)} m from the game camera. Its meshes: ${meshes} triangles.` };
      }
      case "method": {
        const level = choices[step.method][step.i];
        const best = choices.DP?.[step.i];
        let text = `${METHOD_LONG[step.method]} gives ${obj.name} <b>LOD${level}</b>, ${triangles(level)} triangles.`;
        if (step.method !== "DP" && best !== undefined && level !== best) {
          text +=
            level > best
              ? ` The optimal plan gives it LOD${best}: less detail here than it deserves.`
              : ` The optimal plan keeps it at LOD${best}: these triangles are worth more elsewhere.`;
        }
        const q = quality(step.method);
        if (q !== null) text += ` Picture quality <b>${pct(q)}</b> of the best.`;
        return { title: `${obj.name} · ${METHOD_LABEL[step.method]}`, text };
      }
      case "hold": {
        const level = choices.DP[step.i];
        const coarsest = level === obj.triCounts.length - 1;
        return {
          title: obj.name,
          text: `Every plan gives it <b>LOD${level}</b>, ${triangles(level)} triangles${coarsest ? ": more detail here adds too little to the picture to be worth it" : ""}.`,
        };
      }
      default: {
        const scores = METHOD_ORDER.filter((m) => quality(m) !== null)
          .map((m) => `${METHOD_LABEL[m]} <b>${pct(quality(m))}</b>`)
          .join(" · ");
        return { title: "In this frame", text: `${this.data.story ?? ""} Picture quality: ${scores}.` };
      }
    }
  }

  onScreen(i) {
    const p = vec(this.data.scene.objects[i].position).project(this.camera);
    return p.z < 1 && Math.abs(p.x) < 0.85 && Math.abs(p.y) < 0.85;
  }

  screenPoint(i) {
    const p = vec(this.data.scene.objects[i].position).project(this.camera);
    const width = this.wrap.clientWidth;
    const height = this.wrap.clientHeight;
    const clamp = (x, low, high) => Math.min(Math.max(x, low), high);
    return { x: clamp(((p.x + 1) / 2) * width, 20, width - 20), y: clamp(((1 - p.y) / 2) * height, 20, height - 20) };
  }

  // The right end of a button, so the pointer does not hide its label.
  elementPoint(node) {
    const a = node.getBoundingClientRect();
    const b = this.wrap.getBoundingClientRect();
    return { x: a.left - b.left + a.width * 0.78, y: a.top - b.top + a.height * 0.6 };
  }

  placeCursor(x, y) {
    this.cursor.style.transition = "none";
    this.cursor.style.transform = `translate(${x}px, ${y}px)`;
    void this.cursor.offsetWidth;
    this.cursor.style.transition = "";
    this.cursorAt = { x, y };
  }

  aimCursor({ x, y }) {
    this.cursor.classList.remove("lv-cursor-off");
    this.cursor.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    this.cursorAt = { x, y };
  }

  hideCursor() {
    this.cursor?.classList.add("lv-cursor-off");
  }

  // ------------------------------------------------------- LOD choices
  cameraArray() {
    const p = this.mode === "drive" ? this.camera.position : vec(this.data.path[this.data.frame] ?? this.data.path[0]);
    return [p.x, p.y, p.z];
  }

  quantumAt(camera, budget) {
    let best = null;
    for (const entry of this.data.quantum ?? []) {
      if (Math.abs(entry.budget - budget) > 0.5) continue;
      const tolerance = entry.source === "live" ? 0.02 : 0.26;
      const d = this.solver.distance(entry.camera, camera);
      if (d <= tolerance && (best === null || d < best.d)) best = { entry, d };
    }
    return best?.entry ?? null;
  }

  // Recompute the choice of every method for the current view; cheap enough per frame.
  solve() {
    const camera = this.cameraArray();
    const budget = this.mode === "drive" ? this.state.budget : this.data.budget.value;
    const key = `${camera.map((x) => x.toFixed(4)).join(",")}|${budget}|${this.state.method}|${this.data.frame}`;
    if (key === this.lastSolveKey) return false;
    this.lastSolveKey = key;
    const scene = this.data.scene;
    const view = this.solver.solveView(scene.objects, camera, budget, scene.blockSize);
    const triCounts = scene.objects.map((obj) => obj.triCounts);
    const choices = {};
    let quantum = null;
    if (this.mode === "orbit") {
      Object.assign(choices, this.data.choices ?? {});
    } else if (view.feasible) {
      choices.DP = view.dp.choice;
      choices.Greedy = view.greedy.choice;
      quantum = this.quantumAt(camera, budget);
      for (const method of QUANTUM) choices[method] = quantum?.choices?.[method] ?? null;
    }
    const solutions = {};
    for (const [method, choice] of Object.entries(choices)) {
      if (choice && view.feasible) solutions[method] = this.solver.evaluate(choice, view.v, view.dw, triCounts);
    }
    this.current = { camera, budget, view, choices, solutions, quantum };
    this.applyChoice(choices[this.state.method] ?? null);
    return true;
  }

  applyChoice(choice) {
    if (!this.models) return;
    const objects = this.data.scene.objects;
    const changes = [];
    this.models.forEach((model, i) => {
      const level = choice ? choice[i] : null;
      const shown = level ?? objects[i].triCounts.length - 1;
      if (model.shown === null || !this.ready || matchMedia("(prefers-reduced-motion: reduce)").matches) {
        if (model.fade) model.fade = null;
        model.shown = shown;
        model.meshes.forEach((mesh, j) => {
          mesh.visible = j === shown;
          mesh.material.opacity = 1;
        });
      } else if (shown !== model.shown) {
        this.startFade(model, shown);
      }
      if (model.level !== null && level !== null && model.level !== level && this.ready) {
        changes.push({ i, from: model.level, to: level });
      }
      model.level = level;
      this.paint(i);
    });
    this.applyWire();
    // During the tour only the model being inspected gets a toast.
    const touring = this.tour?.state === "playing" && this.state.focused !== null;
    const toasts = touring ? changes.filter((change) => change.i === this.state.focused) : changes;
    if (toasts.length && this.ready) {
      for (const { i, from, to } of toasts.slice(0, 3)) {
        const counts = objects[i].triCounts;
        const delta = counts[to] - counts[from];
        const arrow = to < from ? "more detail" : "less detail";
        this.toast(
          `<span class="lv-dot" style="background:${this.data.lodColors[to]}"></span> <b>${objects[i].name}</b>: LOD${from} → LOD${to} · ${arrow} (${delta > 0 ? "+" : "−"}${fmtInt.format(Math.abs(delta))} triangles)`,
        );
      }
    }
    this.ready = true;
    this.needsRender = true;
  }

  paint(i) {
    const model = this.models[i];
    const color =
      model.level === null
        ? this.theme.gray
        : this.state.lodColors
          ? this.data.lodColors[model.level]
          : this.theme.clay;
    model.meshes.forEach((mesh) => mesh.material.color.set(color));
    model.ring.material.color.set(model.level === null ? this.theme.ink2 : this.data.lodColors[model.level]);
    model.ring.material.opacity = model.level === null ? 0.18 : 0.5;
  }

  toast(html) {
    const node = el("div", "lv-toast", html);
    this.toasts.prepend(node);
    while (this.toasts.children.length > 3) this.toasts.lastChild.remove();
    setTimeout(() => node.remove(), 2700);
  }

  // ---------------------------------------------------------------- HUD
  refresh(force = false) {
    if (this.models && this.solve()) this.hudDirty = true;
    if (force) this.hudDirty = true;
    const now = performance.now();
    // While the camera moves the choice is re-solved every frame, the panels at most ~16 Hz.
    if (!this.hudDirty || (!force && now - this.lastHud < 60)) return;
    this.hudDirty = false;
    this.lastHud = now;
    if (this.mode === "drive") this.renderDriveHud();
    else this.renderOrbitHud();
    this.renderObjects();
    this.renderFocusCard();
  }

  renderOrbitHud() {
    if (!this.legend) return;
    const method = this.state.method;
    const choice = this.data.choices?.[method];
    for (const m of METHOD_ORDER) {
      this.methodButtons[m].setAttribute("aria-pressed", String(m === method));
      this.methodButtons[m].dataset.missing = String(!this.data.choices?.[m]);
    }
    this.legend.querySelector(".lv-frame-text").textContent =
      `Frame ${this.data.frame} · budget ${fmtInt.format(this.data.budget.value)} triangles`;
    const note = this.legend.querySelector(".lv-orbit-note");
    note.textContent = choice ? "" : "No answer for this budget, so the models are gray.";
    note.classList.toggle("lv-hidden", !!choice);
  }

  renderDriveHud() {
    const cur = this.current;
    if (!cur || !this.main) return;
    const { view, solutions } = cur;
    const method = this.state.method;
    for (const m of METHOD_ORDER) this.methodButtons[m].setAttribute("aria-pressed", String(m === method));
    this.main.querySelector(".lv-budget-value").textContent = `${fmtInt.format(cur.budget)} triangles`;

    const t = this.state.pathT;
    const frame = Math.round(t);
    this.main.querySelector(".lv-where").textContent = this.state.onPath
      ? `on the path, frame ${frame}`
      : "free flight";
    if (this.frameText) this.frameText.textContent = this.state.onPath ? `frame ${frame} / ${this.data.path.length - 1}` : "off the path";

    const bar = this.main.querySelector(".lv-used-bar");
    const objects = this.data.scene.objects;
    const maxBudget = Number(this.data.budget.max);
    bar.innerHTML = "";
    const solution = solutions[method];
    if (!view.feasible) {
      this.main.querySelector(".lv-used").textContent = "Budget is smaller than the coarsest models";
      this.main.querySelector(".lv-quality-text").textContent = "";
    } else if (solution) {
      solution.choice.forEach((j, i) => {
        const span = el("span");
        span.style.width = `${(100 * objects[i].triCounts[j]) / maxBudget}%`;
        span.style.background = this.data.lodColors[j];
        span.title = `${objects[i].name}: LOD${j}, ${fmtInt.format(objects[i].triCounts[j])} triangles`;
        bar.appendChild(span);
      });
      this.main.querySelector(".lv-used").textContent = `uses ${fmtInt.format(solution.triangles)} of ${fmtInt.format(cur.budget)}`;
      this.main.querySelector(".lv-quality-text").textContent = `quality ${pct(solution.value / view.dp.value)} of best`;
    } else {
      this.main.querySelector(".lv-used").textContent = `${METHOD_LONG[method]} has no answer for this view`;
      this.main.querySelector(".lv-quality-text").textContent = "models shown in gray";
    }
    const limit = el("span", "lv-limit");
    limit.style.left = `${(100 * cur.budget) / maxBudget}%`;
    limit.title = "Budget";
    bar.appendChild(limit);

    for (const m of METHOD_ORDER) {
      const cell = this.scoreCells[m];
      const s = solutions[m];
      cell.dataset.active = String(m === method);
      cell.querySelector(".lv-score-value").textContent = s && view.feasible ? pct(s.value / view.dp.value) : "–";
      let note = "";
      if (m === "DP") note = "the referee";
      else if (m === "Greedy") note = s && s.value < view.dp.value - 1e-12 ? "falls short" : "matches";
      else if (s) note = cur.quantum?.source === "live" ? "run here" : `frame ${cur.quantum?.frame}`;
      else note = "not run here";
      cell.querySelector(".lv-score-note").textContent = note;
    }
    this.renderQuantumBox();
    this.drawMinimap();
  }

  renderQuantumBox() {
    const box = this.quantumBox;
    const cur = this.current;
    const quantumSelected = QUANTUM.has(this.state.method);
    const missing = !cur.quantum;
    const pending = this.state.pending;
    if (!pending && !(quantumSelected && missing)) {
      box.classList.add("lv-hidden");
      return;
    }
    box.classList.remove("lv-hidden");
    if (pending) {
      box.innerHTML =
        '<div class="lv-row lv-small"><span><span class="lv-spinner lv-inline"></span>Running QAOA with the X and XY mixers on the simulated quantum computer…</span></div>';
      return;
    }
    box.innerHTML = `<div class="lv-small">QAOA takes a few seconds per view, so it is not recomputed while you fly. It has answers along the precomputed path and for every view you run; this view has none yet.</div>`;
    const run = button("Run QAOA for this view", { className: "lv-btn lv-primary" });
    run.onclick = () => this.requestQuantum();
    box.appendChild(run);
  }

  requestQuantum() {
    const cur = this.current;
    if (!cur || !cur.view.feasible) return;
    const camera = cur.camera.map((x) => Math.round(x * 1e4) / 1e4);
    this.state.pending = { camera, budget: cur.budget };
    this.setPlaying(false);
    this.trigger("run_qaoa", { camera, budget: cur.budget, frame: Math.round(this.state.pathT) });
    this.renderQuantumBox();
  }

  renderObjects() {
    if (!this.objectRows || !this.current) return;
    const objects = this.data.scene.objects;
    const { view } = this.current;
    const levels = this.models?.map((m) => m.level) ?? [];
    if (!this.objectCells) {
      this.objectRows.innerHTML = "";
      this.objectCells = objects.map((obj, i) => {
        const row = el("tr", "lv-obj");
        row.innerHTML = `<td><b>${obj.name}</b></td><td class="lv-lodcell"></td><td class="lv-num lv-tris"></td><td class="lv-num lv-dist"></td><td></td>`;
        const go = button("", { icon: "target", title: `Fly to ${obj.name}`, className: "lv-btn lv-icon" });
        go.style.height = "24px";
        go.style.width = "24px";
        row.lastChild.appendChild(go);
        row.onclick = () => this.focus(i);
        this.objectRows.appendChild(row);
        return { row, lod: row.querySelector(".lv-lodcell"), tris: row.querySelector(".lv-tris"), dist: row.querySelector(".lv-dist"), level: undefined };
      });
    }
    this.objectCells.forEach((cell, i) => {
      const level = levels[i];
      if (cell.level !== level) {
        cell.lod.innerHTML =
          level === null || level === undefined
            ? '<span class="lv-lod">–</span>'
            : `<span class="lv-lod"><span class="lv-dot" style="background:${this.data.lodColors[level]}"></span>LOD${level}</span>`;
        cell.tris.textContent = level === null || level === undefined ? "–" : fmtK(objects[i].triCounts[level]);
        if (cell.level !== undefined) {
          cell.row.classList.remove("lv-changed");
          void cell.row.offsetWidth;
          cell.row.classList.add("lv-changed");
        }
        cell.level = level;
      }
      cell.dist.textContent = `${view.distances[i].toFixed(1)} m`;
      cell.row.dataset.focused = String(this.state.focused === i);
    });
  }

  renderFocusCard() {
    const i = this.state.focused;
    if (i === null || !this.current || !this.models) {
      this.focusCard.classList.add("lv-hidden");
      return;
    }
    const obj = this.data.scene.objects[i];
    const level = this.models[i].level;
    const distance = this.solver.distance(obj.position, [this.camera.position.x, this.camera.position.y, this.camera.position.z]);
    const meshTris = this.models[i].meshes.map((mesh) => mesh.geometry.userData.triangles);
    const rows = obj.triCounts
      .map(
        (count, j) =>
          `<div class="lv-level" data-active="${j === level}"><span><span class="lv-dot" style="background:${this.data.lodColors[j]}"></span> LOD${j}</span><span>${fmtInt.format(count)}</span></div>`,
      )
      .join("");
    const why =
      this.mode === "drive"
        ? "Closer models fill more of the screen, so detail on them is worth more. The budget decides how much detail the whole frame can afford."
        : "Showing every triangle of the level this method picked.";
    this.focusCard.innerHTML = `
      <div class="lv-row"><strong>${obj.name}</strong><span class="lv-small lv-muted">${distance.toFixed(1)} m from you</span></div>
      <div class="lv-levels">${rows}</div>
      <div class="lv-small lv-muted">Budget counts per level (triangles). The mesh on screen has ${fmtInt.format(meshTris[level ?? meshTris.length - 1])}.</div>
      <div class="lv-small lv-muted">${why}</div>`;
    const leave = button("Leave focus", { className: "lv-btn" });
    leave.onclick = () => this.clearFocus();
    this.focusCard.appendChild(leave);
    this.focusCard.classList.remove("lv-hidden");
  }

  drawMinimap() {
    const canvas = this.minimap;
    if (!canvas || !this.center) return;
    const size = 190;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== size * ratio) {
      canvas.width = size * ratio;
      canvas.height = size * ratio;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const objects = this.data.scene.objects;
    const cam = this.camera.position;
    const xs = [...objects.flatMap((o) => [o.position[0] - o.radius, o.position[0] + o.radius]), ...this.data.path.map((p) => p[0]), cam.x];
    const zs = [...objects.flatMap((o) => [o.position[2] - o.radius, o.position[2] + o.radius]), ...this.data.path.map((p) => p[2]), cam.z];
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    const span = Math.max(maxX - minX, maxZ - minZ, 1) * 1.12;
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    const scale = size / span;
    // x to the right, z (forward) up the map.
    const sx = (x) => size / 2 + (x - cx) * scale;
    const sy = (z) => size / 2 - (z - cz) * scale;
    const ink = this.theme.ink;
    const ink2 = this.theme.ink2;

    const heading = new THREE.Vector3();
    this.camera.getWorldDirection(heading);
    const angle = Math.atan2(heading.z, heading.x);
    const half = Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * this.camera.aspect);
    const reach = span * 1.5;
    ctx.fillStyle = this.data.theme === "dark" ? "rgba(255,255,255,0.08)" : "rgba(11,11,11,0.07)";
    ctx.beginPath();
    ctx.moveTo(sx(cam.x), sy(cam.z));
    ctx.lineTo(sx(cam.x + reach * Math.cos(angle - half)), sy(cam.z + reach * Math.sin(angle - half)));
    ctx.lineTo(sx(cam.x + reach * Math.cos(angle + half)), sy(cam.z + reach * Math.sin(angle + half)));
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = ink2;
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    this.data.path.forEach((p, k) => (k ? ctx.lineTo(sx(p[0]), sy(p[2])) : ctx.moveTo(sx(p[0]), sy(p[2]))));
    ctx.stroke();
    ctx.setLineDash([]);

    const order = objects.map((o, i) => i).sort((a, b) => objects[b].radius - objects[a].radius);
    for (const i of order) {
      const o = objects[i];
      const level = this.models?.[i]?.level;
      ctx.beginPath();
      ctx.arc(sx(o.position[0]), sy(o.position[2]), Math.max(o.radius * scale, 3), 0, Math.PI * 2);
      ctx.fillStyle = level === null || level === undefined ? this.theme.gray : this.data.lodColors[level];
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = this.data.theme === "dark" ? "#1c1f24" : "#fcfcfb";
      ctx.stroke();
    }
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = ink;
    ctx.textAlign = "center";
    for (const o of objects) ctx.fillText(o.name, sx(o.position[0]), sy(o.position[2]) - Math.max(o.radius * scale, 3) - 3);

    ctx.save();
    ctx.translate(sx(cam.x), sy(cam.z));
    ctx.rotate(-angle + Math.PI / 2);
    ctx.beginPath();
    ctx.moveTo(0, -7);
    ctx.lineTo(5, 5);
    ctx.lineTo(-5, 5);
    ctx.closePath();
    ctx.fillStyle = ink;
    ctx.fill();
    ctx.restore();
  }

  updateLabels() {
    if (!this.models || !this.state.labels) return;
    const width = this.wrap.clientWidth;
    const height = this.wrap.clientHeight;
    const objects = this.data.scene.objects;
    const p = new THREE.Vector3();
    const camPos = this.camera.position;
    const placed = [];
    this.models.forEach((model, i) => {
      const obj = objects[i];
      p.set(obj.position[0], obj.position[1] + obj.radius * 1.12, obj.position[2]).project(this.camera);
      const visible = p.z < 1 && Math.abs(p.x) < 1.2 && Math.abs(p.y) < 1.2;
      model.label.style.opacity = visible ? "1" : "0";
      model.label.style.pointerEvents = visible ? "auto" : "none";
      if (!visible) return;
      placed.push({ model, x: ((p.x + 1) / 2) * width, y: ((1 - p.y) / 2) * height, depth: p.z });
      const level = model.level;
      const distance = camPos.distanceTo(vec(obj.position));
      const key = `${level}|${distance.toFixed(1)}|${this.mode}`;
      if (model.labelKey === key) return;
      model.labelKey = key;
      const lod =
        level === null
          ? '<span class="lv-lod">no answer</span>'
          : `<span class="lv-lod"><span class="lv-dot" style="background:${this.data.lodColors[level]}"></span>LOD${level}</span>`;
      const extra = level === null ? "" : `<span class="lv-sub">${fmtK(obj.triCounts[level])} △</span>`;
      const dist = this.mode === "drive" ? `<span class="lv-sub">${distance.toFixed(1)} m</span>` : "";
      model.label.innerHTML = `<b>${obj.name}</b>${lod}${extra}${dist}`;
      model.labelSize = null;
    });
    // The nearest model keeps its label in place; others move up until they no longer overlap.
    placed.sort((a, b) => a.depth - b.depth);
    const boxes = [];
    for (const item of placed) {
      const label = item.model.label;
      item.model.labelSize ??= { w: label.offsetWidth, h: label.offsetHeight };
      const { w, h } = item.model.labelSize;
      let top = item.y - h;
      for (let k = 0; k < 8; k += 1) {
        const hit = boxes.find((b) => item.x - w / 2 < b.right && item.x + w / 2 > b.left && top < b.bottom && top + h > b.top);
        if (!hit) break;
        top = hit.top - h - 3;
      }
      boxes.push({ left: item.x - w / 2, right: item.x + w / 2, top, bottom: top + h });
      label.style.transform = `translate(${(item.x - w / 2).toFixed(1)}px, ${top.toFixed(1)}px)`;
    }
  }

  // --------------------------------------------------------------- loop
  loop(now) {
    if (this.disposed) return;
    this.frameId = requestAnimationFrame(this.loop);
    const elapsed = now - this.clock;
    const dt = Math.min(elapsed / 1000, 0.1);
    this.clock = now;
    // While the view is off screen or the tab sleeps, the tour clock stands still.
    if (this.tour?.state === "playing" && (elapsed > 1000 || !this.visible || document.hidden)) this.tour.start += elapsed;
    if (!this.visible || document.hidden) return;

    if (this.tween) {
      const t = Math.min((now - this.tween.start) / this.tween.duration, 1);
      const e = easeInOut(t);
      this.camera.position.lerpVectors(this.tween.fromPos, this.tween.toPos, e);
      this.controls.target.lerpVectors(this.tween.fromTarget, this.tween.toTarget, e);
      if (t >= 1) this.tween = null;
      this.needsRender = true;
    }
    if (this.gizmoTween) {
      const g = this.gizmoTween;
      const t = Math.min((now - g.start) / g.duration, 1);
      this.gizmo.position.lerpVectors(g.from, g.to, easeInOut(t));
      this.gizmo.lookAt(g.target);
      if (t >= 1) this.gizmoTween = null;
      this.needsRender = true;
    }
    if (this.mode === "drive") this.drive(dt);
    else this.stepTour(now);
    // The tour slowly circles whatever it looks at.
    this.controls.autoRotate = this.tour?.state === "playing" && !this.tween && !this.reducedMotion.matches;
    this.controls.autoRotateSpeed = 0.8;
    this.controls.update(dt);

    if (this.models) this.stepFades(now);

    this.refresh();
    if (this.needsRender) {
      this.needsRender = false;
      this.renderer.render(this.scene, this.camera);
      this.updateLabels();
    }
  }

  drive(dt) {
    if (this.state.playing) {
      const last = this.data.path.length - 1;
      const t = this.state.pathT + dt * 2.5;
      if (t >= last) {
        this.goToPath(last, false);
        this.setPlaying(false);
      } else this.goToPath(t, false);
    }
    if (!this.keys.size) return;
    const fast = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 3 : 1;
    const speed = this.radius * 0.22 * fast * dt;
    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    const right = new THREE.Vector3().crossVectors(forward, this.camera.up).normalize();
    const move = new THREE.Vector3();
    if (this.keys.has("KeyW")) move.add(forward);
    if (this.keys.has("KeyS")) move.sub(forward);
    if (this.keys.has("KeyD")) move.add(right);
    if (this.keys.has("KeyA")) move.sub(right);
    if (this.keys.has("KeyE")) move.y += 1;
    if (this.keys.has("KeyQ")) move.y -= 1;
    if (move.lengthSq() === 0) return;
    move.normalize().multiplyScalar(speed);
    this.camera.position.add(move);
    this.controls.target.add(move);
    this.state.onPath = false;
    this.needsRender = true;
  }

  fail(error) {
    this.status.classList.remove("lv-hidden");
    this.status.innerHTML = `<div style="max-width:420px;text-align:center">The 3D view could not start: ${String(error.message ?? error)}</div>`;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.frameId);
    this.resizeObserver.disconnect();
    this.intersection.disconnect();
    if (this.onWindowResize) window.removeEventListener("resize", this.onWindowResize);
    this.barObserver?.disconnect();
    this.controls.dispose();
    this.scene.traverse((node) => {
      if (node.material && node.material !== this.wireMaterial) node.material.dispose?.();
    });
    this.wireMaterial?.dispose();
    this.renderer.dispose();
    this.wrap.remove();
  }
}

const instances = new Map();

export async function mount(component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component;
  const solver = await solverReady;
  let entry = instances.get(parentElement);
  if (entry) clearTimeout(entry.timer);
  if (entry && !entry.viewer.update(data)) {
    entry.viewer.dispose();
    entry = null;
  }
  if (!entry) {
    entry = { viewer: new LodViewer(parentElement, data, solver), timer: null };
    instances.set(parentElement, entry);
  }
  entry.viewer.trigger = setTriggerValue;
  entry.viewer.setState = setStateValue;
  return () => {
    // Streamlit calls this before every update as well as on unmount; only an unmount
    // that is not followed by a new mount within a moment disposes the viewer.
    entry.timer = setTimeout(() => {
      entry.viewer.dispose();
      instances.delete(parentElement);
    }, 1500);
  };
}
