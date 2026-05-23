// 彩种配置中心
//
// 每个彩种把"号码空间 / 字段名 / 业务规则 / 视觉常量"集中暴露，让通用算法
// 无需关心是双色球还是大乐透。
//
// 双色球字段：{ issue, date, reds: [6 个 1-33], blue: 1-16 }
// 大乐透字段：{ issue, date, front: [5 个 1-35], back: [2 个 1-12] }

export const SSQ_CONFIG = {
  id: "ssq",
  name: "双色球",
  enName: "SSQ",
  zones: [
    { key: "reds", label: "红球", color: "red", size: 33, pick: 6, isArray: true, max: 33 },
    { key: "blue", label: "蓝球", color: "blue", size: 16, pick: 1, isArray: false, max: 16 },
  ],
  pricePerTicket: 2,
  // 一等奖中奖概率 = 1 / [C(33,6) × 16] = 1/17,721,088
  jackpotProbability: 1 / 17721088,
  jackpotDenominator: 17721088,
  // 开奖时刻（中国时区）
  drawDays: [0, 2, 4], // 周二/四/日
  drawHour: 21,
  drawMinute: 15,
  saleCutoffHour: 20,
  saleCutoffMinute: 0,
  // 期号格式：7 位（2026054）
  issuePadLength: 7,
};

export const DLT_CONFIG = {
  id: "dlt",
  name: "大乐透",
  enName: "DLT",
  zones: [
    { key: "front", label: "前区", color: "front", size: 35, pick: 5, isArray: true, max: 35 },
    { key: "back",  label: "后区", color: "back",  size: 12, pick: 2, isArray: true, max: 12 },
  ],
  pricePerTicket: 2,
  // 一等奖中奖概率 = 1 / [C(35,5) × C(12,2)] = 1/21,425,712
  jackpotProbability: 1 / 21425712,
  jackpotDenominator: 21425712,
  // 开奖时刻（中国时区）
  drawDays: [1, 3, 6], // 周一/三/六
  drawHour: 20,
  drawMinute: 30,
  saleCutoffHour: 20,
  saleCutoffMinute: 0,
  // 期号格式：5 位（26054）—— 大乐透官方/500.com 都是这种短格式
  issuePadLength: 5,
};

/** 取彩种配置（id 或对象都接受）。 */
export function resolveLotteryConfig(idOrConfig) {
  if (typeof idOrConfig === "object" && idOrConfig?.id) return idOrConfig;
  if (idOrConfig === "ssq") return SSQ_CONFIG;
  if (idOrConfig === "dlt") return DLT_CONFIG;
  throw new Error(`unknown lottery: ${idOrConfig}`);
}

/** 把单期开奖按"区域"取出号码列表（统一返回数组形式）。 */
export function getZoneNumbers(draw, zone) {
  const v = draw[zone.key];
  if (zone.isArray) return Array.isArray(v) ? v : [];
  return v == null ? [] : [v];
}

/** 双色球与大乐透通用：把开奖记录归一化为 { issue, date, [zoneKey]: ... } */
export function normaliseDraw(raw, config) {
  const out = { issue: String(raw.issue), date: raw.date || null };
  if (raw.year != null) out.year = raw.year;
  for (const z of config.zones) {
    const v = raw[z.key];
    if (z.isArray) {
      out[z.key] = Array.isArray(v) ? v.slice().sort((a, b) => a - b) : [];
    } else {
      out[z.key] = Number.isFinite(v) ? v : null;
    }
  }
  return out;
}

/** 校验一期开奖是否合法。 */
export function validateDraw(draw, config) {
  for (const z of config.zones) {
    const nums = getZoneNumbers(draw, z);
    if (nums.length !== z.pick) return `${z.label}应为 ${z.pick} 个`;
    const set = new Set(nums);
    if (set.size !== z.pick) return `${z.label}有重复`;
    for (const n of nums) {
      if (!Number.isInteger(n) || n < 1 || n > z.max) {
        return `${z.label} ${n} 超出 1-${z.max}`;
      }
    }
  }
  return null;
}
