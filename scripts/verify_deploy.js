/* Definitive check: does the LIVE GitHub Pages demo complete inference?
   GitHub Pages cannot set COOP/COEP, and the demo uses onnxruntime-web's threaded
   WASM build, which normally requires SharedArrayBuffer. So this drives real Chrome
   and reports console output + the page's own result, rather than assuming. */
const puppeteer = require("puppeteer-core");

const URL = process.argv[2] || "https://kcheung-inno.github.io/chestxray14-lesion-detector/";
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push("[console." + m.type() + "] " + m.text()));
  page.on("pageerror", (e) => logs.push("[pageerror] " + e.message));
  page.on("requestfailed", (r) =>
    logs.push("[requestfailed] " + r.url().slice(0, 110) + " :: " + (r.failure() || {}).errorText));

  console.log("navigating:", URL);
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 120000 });

  const env = await page.evaluate(() => ({
    sab: typeof SharedArrayBuffer,
    isolated: self.crossOriginIsolated,
    webgpu: !!navigator.gpu,
  }));
  console.log("environment:", JSON.stringify(env));

  // wait for the page to reach a terminal state
  let last = "";
  for (let i = 0; i < 60; i++) {
    const t = await page.evaluate(() => {
      const e = document.getElementById("timing");
      return e ? e.textContent : "no-el";
    });
    if (t !== last) { console.log(`[${i * 10}s] timing: ${t}`); last = t; }
    if (/analysed in|failed/.test(t)) break;
    await new Promise((r) => setTimeout(r, 10000));
  }

  console.log("--- result ---");
  console.log(await page.evaluate(() => JSON.stringify({
    timing: document.getElementById("timing")?.textContent,
    gt: document.getElementById("sGt")?.textContent,
    preds: document.getElementById("sPred")?.textContent,
    hit: document.getElementById("sHit")?.textContent,
    noPred: document.getElementById("noPred")?.textContent,
    rows: Array.from(document.querySelectorAll("#tbl tbody tr")).map(
      (tr) => Array.from(tr.children).map((td) => td.textContent.trim()).join(" | ")),
  })));

  console.log("--- console / errors ---");
  console.log(logs.length ? logs.slice(-25).join("\n") : "(none)");
  await browser.close();
})().catch((e) => { console.error("HARNESS ERROR:", e.message); process.exit(1); });
