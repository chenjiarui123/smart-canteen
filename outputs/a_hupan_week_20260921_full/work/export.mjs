import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

const work = path.dirname(fileURLToPath(import.meta.url));
const entries = JSON.parse(await fs.readFile(path.join(work, 'week.json'), 'utf8'));
const order = ['荤菜', '素菜', '主食', '甜品', '水果', '其他'];
const quote = v => '"' + String(v ?? '').replaceAll('"', '""') + '"';
const report = [];
for (const entry of entries) {
  const rows = [['推荐来源','菜品种类','菜品ID','菜品名称','预测销售额（元）','预测销量','销量单位']];
  for (const [source, items] of [['常用菜', entry.high], ['低频潜力菜', entry.low]]) {
    const categories = [...new Set(items.map(i => i.category))].sort((a,b) =>
      (order.includes(a) ? order.indexOf(a) : 99) - (order.includes(b) ? order.indexOf(b) : 99) || a.localeCompare(b, 'zh'));
    for (const category of categories) {
      rows.push([source, category, '', '', '', '', '']);
      const group = items.filter(i => i.category === category).sort((a,b) =>
        b.reference_revenue - a.reference_revenue || String(a.dish_id).localeCompare(String(b.dish_id)));
      for (const item of group) {
        rows.push([source, item.category, String(item.dish_id), item.dish_name,
          Math.round(item.reference_revenue*100)/100, Math.round(item.quantity*100)/100, item.unit]);
      }
    }
  }
  const wb = Workbook.create();
  const sheetName = entry.weekday + entry.meal;
  const sheet = wb.worksheets.add(sheetName);
  sheet.getRange(`A1:G${rows.length}`).values = rows;
  sheet.getRange(`A1:G${rows.length}`).format.font = {name:'Arial',size:11};
  sheet.getRange(`A1:G${rows.length}`).format.rowHeight = 24;
  for (const [col,width] of [['A',18],['B',14],['C',14],['D',30],['E',23],['F',20],['G',14]]) {
    sheet.getRange(`${col}1`).format.columnWidth = width;
  }
  sheet.getRange(`C2:C${rows.length}`).setNumberFormat('@');
  sheet.getRange(`E2:F${rows.length}`).setNumberFormat('0.00');
  wb.recalculate();
  const values = sheet.getRange(`A1:G${rows.length}`).values;
  if (JSON.stringify(values) !== JSON.stringify(rows)) throw Error('Cell values changed: ' + sheetName);
  const inspection = await wb.inspect({kind:'table',range:`${sheetName}!A1:G4`,include:'values',tableMaxRows:4,tableMaxCols:7});
  report.push({file:sheetName,inspection:inspection.ndjson});
  await fs.writeFile(path.join(work,`../menus/A湖畔_${entry.weekday}_${entry.meal}.csv`),
    '\uFEFF'+values.map(row => row.map(quote).join(',')).join('\r\n')+'\r\n');
  const preview = await wb.render({sheetName,range:'A1:G10',scale:1,format:'png'});
  await fs.writeFile(path.join(work,`${sheetName}_preview.png`),new Uint8Array(await preview.arrayBuffer()));
  console.log(`${sheetName}: ${entry.high.length} + ${entry.low.length}`);
}
await fs.writeFile(path.join(work,'artifact-inspection.json'),JSON.stringify(report,null,2));
