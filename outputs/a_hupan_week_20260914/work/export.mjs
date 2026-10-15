import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook} from '@oai/artifact-tool';

const work=path.dirname(fileURLToPath(import.meta.url));
const out=path.dirname(work);
const sample=process.argv.includes('--sample');
const entries=JSON.parse(await fs.readFile(path.join(work,sample?'sample.json':'week.json'),'utf8'));
const headers=['行类型','推荐来源','菜品种类','菜品ID','菜品名称','预测销售额（元）','预测销量','销量单位','数量及金额口径'];
const order=['荤菜','素菜','主食','甜品','水果','其他'];
const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
for(const entry of entries){
 const wb=Workbook.create(); const sh=wb.worksheets.add('菜单');
 const row=(a)=>[...a,...Array(Math.max(0,9-a.length)).fill('')];
 const rows=[row([`A湖畔智慧餐厅 ${entry.date} ${entry.weekday} ${entry.meal}`]),
  row(['情景预测：普通校历；天气取2025年9月30天中位数，非目标日实测或预报']),
  row([`气温${entry.weather.temperature_c}℃；湿度${entry.weather.humidity_pct}%；降水${entry.weather.precipitation_mm}毫米；风速${entry.weather.wind_speed_kmh}公里/小时`]),
  row([`真实销售历史截至${entry.audit.history_end}；常用菜${entry.high.length}道；低频菜${entry.low.length}/24道`]),
  row(['低频菜为备选：预测销量列列示试餐参考量，销售额列为按试餐量估计的参考额；不与常用菜直接合并备餐']),
  headers];
 const bands=[];
 for(const [source,items] of [['常用菜',entry.high],['低频潜力菜',entry.low]]){
  const categories=[...new Set(items.map(i=>i.category))].sort((a,b)=>(order.indexOf(a)<0?99:order.indexOf(a))-(order.indexOf(b)<0?99:order.indexOf(b))||a.localeCompare(b,'zh'));
  for(const category of categories){
   bands.push(rows.length+1);rows.push(row(['类别分组',source,category]));
   const group=items.filter(i=>i.category===category).sort((a,b)=>b.reference_revenue-a.reference_revenue||a.dish_id.localeCompare(b.dish_id));
   for(const i of group) rows.push(['菜品',source,i.category,String(i.dish_id),i.dish_name,Math.round(i.reference_revenue*100)/100,Math.round(i.quantity*100)/100,i.unit,source==='常用菜'?'需求预测':'试餐量及参考销售额']);
  }
 }
 if(entry.low.length<24) rows.push(row([`低频候选不足：当前策略仅找到${entry.low.length}道合格菜，缺${24-entry.low.length}道；未放宽筛选条件或重复凑数`]));
 sh.getRange(`A1:I${rows.length}`).values=rows;
 sh.getRange(`A1:I${rows.length}`).format.font={name:'Arial',size:11};
 sh.getRange(`A1:I${rows.length}`).format.rowHeight=23;
 sh.getRange('A1:I1').format.font={bold:true,size:14};
 sh.getRange('A6:I6').format={fill:'#253B52',font:{bold:true,color:'#FFFFFF'}};
 sh.getRange('A1:A5').format.columnWidth=16;
 sh.getRange('B1:C1').format.columnWidth=17;
 sh.getRange('D1:D1').format.columnWidth=18;
 sh.getRange('E1:E1').format.columnWidth=25;
 sh.getRange('F1:G1').format.columnWidth=21;
 sh.getRange('H1:H1').format.columnWidth=12;
 sh.getRange('I1:I1').format.columnWidth=26;
 sh.getRange(`D7:D${rows.length}`).setNumberFormat('@');
 sh.getRange(`F7:G${rows.length}`).setNumberFormat('0.00');
 sh.showGridLines=false;
 for(const r of bands)sh.getRange(`A${r}:I${r}`).format={fill:'#E7EDF4',font:{bold:true}};
 wb.recalculate();
 const values=sh.getRange(`A1:I${rows.length}`).values;
 if(JSON.stringify(values)!==JSON.stringify(rows))throw Error('Cell values changed during export');
 const filename=`A湖畔_${entry.date}_${entry.weekday}_${entry.meal}.csv`;
 await fs.writeFile(path.join(out,'menus',filename),'\uFEFF'+values.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n');
 const preview=await wb.render({sheetName:'菜单',range:'A1:I19',scale:1,format:'png'});
 await fs.writeFile(path.join(work,`${entry.date}_${entry.meal}_preview.png`),new Uint8Array(await preview.arrayBuffer()));
 console.log(JSON.stringify({file:filename,high:entry.high.length,low:entry.low.length,rows:rows.length}));
}
