import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { SwarmController, scenarioSpawns } from '../vendor/microduck-simulator/app/src/game/swarm-controller.js';

const sources=['src/lib/swarm-contract.ts','src/lib/swarm-jev.ts','vendor/microduck-simulator/app/src/game/swarm-controller.js'];
const report={timestamp:new Date().toISOString(),scope:'Live Jev decisions from native scenarioSpawns and SwarmController.status, with standing legged readiness. No physics rollout is performed by this script.',sourceSha256:Object.fromEntries(sources.map(path=>[path,crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')])),cases:[]};
const out=process.argv[2]||'output/swarm-starts.json';
fs.mkdirSync(path.dirname(out), {recursive:true});
const modes=process.argv.slice(3);const scenarios=modes.length?modes:['flock','flock','flock','gather','convoy','split'];
for(const [index,scenario] of scenarios.entries()){
  const poses=scenarioSpawns(scenario).map(pose=>({...pose,loco:'legs',posture:'standing',fallen:false,paused:false,busy:false,manual:false}));
  const controller=new SwarmController();controller.startScenario(scenario,`native-start-${index}`,poses);
  const request={state:{seq:20,time:2,ready:true,paused:false,swarm:controller.status(poses)},memory:{recent:[]}};
  if(index>0&&scenario==='flock')request.memory.recent=Array.from({length:Math.min(index,6)},()=>({intent:'hold',outcome:'complete',progressM:0,targetErrorBeforeM:null,targetErrorAfterM:null,minSeparationM:request.state.swarm.minSeparationM}));
  const response=await fetch('http://127.0.0.1:3177/api/swarm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
  const result=await response.json();report.cases.push({scenario,request,status:response.status,result});fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({scenario,holds:request.memory.recent.length,status:response.status,spread:request.state.swarm.spreadM,intent:result.intent,confidence:result.confidence,abstained:result.abstained,alternatives:result.alternatives}));
}
