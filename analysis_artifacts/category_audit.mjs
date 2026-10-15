import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const paths = process.argv.slice(2);
if (paths[0] === "--csv-help") {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Sheet1");
  sheet.getRange("A1").values = [["x"]];
  console.log(workbook.help("CSV export", {
    search: "csv|delimited|text export",
    include: "index,examples,notes",
    maxChars: 5000,
  }).ndjson);
  process.exit(0);
}
for (const path of paths) {
  const csvText = await fs.readFile(path, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "菜品库" });
  const inspected = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 2500,
    tableMaxRows: 6,
    tableMaxCols: 8,
  });
  console.log(`FILE=${path}`);
  console.log(inspected.ndjson);
  const preview = await workbook.render({
    sheetName: "菜品库",
    range: "A1:H18",
    scale: 1,
    format: "png",
  });
  const outputPath = `analysis_artifacts/${path.includes("长春A") ? "a" : "f"}_preview.png`;
  await fs.writeFile(outputPath, new Uint8Array(await preview.arrayBuffer()));
  console.log(`PREVIEW=${outputPath}`);
}
