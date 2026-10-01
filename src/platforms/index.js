// Platform registry. A platform = estimation Model + truth dynamics + sensors + autopilot + rules.
import { roverPlatform } from './rover.js';
import { jetPlatform } from './jet.js';
import { spacecraftPlatform } from './spacecraft.js';

const reg = new Map([roverPlatform, jetPlatform, spacecraftPlatform].map((p) => [p.id, p]));
export const getPlatform = (id) => reg.get(id);
export const listPlatforms = () => [...reg.values()];
