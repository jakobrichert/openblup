// Browser tests for OpenBLUP Studio (Playwright + Chromium).
//
//   npm install --prefix studio/tests --no-save playwright-core
//   node studio/tests/node_modules/playwright-core/cli.js install chromium   # or set CHROME_PATH
//   python3 -m http.server 8765 --directory studio &
//   node studio/tests/e2e.mjs
//
// Every example is fitted and every result tab rendered; in the 3D tab the
// scene switcher is exercised back and forth. Fails on console errors.
import { chromium } from "playwright-core";

const base = process.env.STUDIO_URL || "http://127.0.0.1:8765/";
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || undefined,
  headless: true,
  // Software WebGL where there is no GPU (CI).
  args: process.env.GPU ? ["--use-angle=metal"] : ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const problems = [];
page.on("console", (m) => {
  // Software GL reports its own performance warnings; they are not ours.
  if (m.type() === "error" || (m.type() === "warning" && !/GPU stall|swiftshader|WebGL/i.test(m.text()))) {
    problems.push(`console ${m.type()}: ${m.text()}`);
  }
});
page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));

const ready = () => page.waitForSelector("html[data-studio-ready]", { timeout: 60000 });
const stageReady = () => page.waitForFunction(() => document.querySelector(".stage-status")?.hidden, null, { timeout: 90000 });
const scene3d = () => page.evaluate(() => ({
  pressed: [...document.querySelectorAll('[aria-label="3D scene"] button')]
    .filter((b) => b.getAttribute("aria-pressed") === "true").map((b) => b.textContent),
  caption: document.querySelector(".stage-caption strong")?.textContent || "",
  canvases: document.querySelectorAll(".stage canvas").length,
}));

const captions = {
  "Field landscape": /observed|trend/i,
  "Likelihood landscape": /likelihood/i,
  "Pedigree": /pedigree/i,
  "G×E landscape": /reaction norms/i,
};
const firstScene = { spatial: "Field landscape", met: "G×E landscape", animal: "Pedigree", rcbd: "Likelihood landscape" };

for (const example of ["spatial", "met", "animal", "rcbd"]) {
  await page.goto(`${base}?example=${example}`);
  await ready();
  const error = await page.$eval(".status.error", (e) => e.textContent).catch(() => null);
  if (error) problems.push(`${example}: fit failed: ${error}`);
  const tabs = await page.$$eval(".tabs button", (bs) => bs.map((b) => b.id.replace("tab-", "")));
  for (const tab of tabs) {
    await page.click(`#tab-${tab}`);
    if (tab === "3d") await stageReady();
    await page.waitForTimeout(150);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
    if (overflow) problems.push(`${example}/${tab}: horizontal page overflow`);
  }

  // 3D scene switching
  await page.click("#tab-3d");
  await stageReady();
  const first = firstScene[example];
  const sequence = first === "Likelihood landscape" ? [] : ["Likelihood landscape", first, "Likelihood landscape", first];
  const expect = async (label, wanted) => {
    await stageReady();
    const s = await scene3d();
    if (s.pressed.length !== 1 || s.pressed[0] !== wanted || !captions[wanted].test(s.caption) || s.canvases !== 1) {
      problems.push(`${example} 3D ${label}: expected ${wanted}, got ${JSON.stringify(s)}`);
    }
  };
  await expect("initial", first);
  for (const [i, name] of sequence.entries()) {
    await page.click(`[aria-label="3D scene"] button:has-text("${name}")`);
    await expect(`step ${i + 1}`, name);
  }
  for (const name of [...sequence, ...sequence]) {
    await page.click(`[aria-label="3D scene"] button:has-text("${name}")`);
    await page.waitForTimeout(40);
  }
  if (sequence.length) await expect("after rapid switching", sequence[sequence.length - 1]);
  console.log(`ok ${example}: tabs ${tabs.join(", ")}`);
}

await browser.close();
if (problems.length) {
  console.error(`FAILED:\n${[...new Set(problems)].join("\n")}`);
  process.exit(1);
}
console.log("all browser checks passed");
