// Filter registry. Register your own class here (or paste one into the in-app Filter Lab).
import { DeadReckoning } from './deadreckon.js';
import { EKF } from './ekf.js';
import { UKF } from './ukf.js';
import { ParticleFilter } from './pf.js';

const registry = new Map();
export const registerFilter = (cls) => registry.set(cls.id, cls);
export const unregisterFilter = (id) => registry.delete(id);
export const getFilterClass = (id) => registry.get(id);
export const listFilters = () => [...registry.values()];

[DeadReckoning, EKF, UKF, ParticleFilter].forEach(registerFilter);
export { DeadReckoning, EKF, UKF, ParticleFilter };
