// 走势图右侧标准列：出现次数 / 平均遗漏 / 最大遗漏 / 当前遗漏
//
// "遗漏期数" 指相邻两次出现之间未出现的期数。例如号码 7 的出现序列为
// 第 [3, 10, 12] 期，则间隔为 [3, 7, 2]（3=从开始到首现; 7,2=两次相邻出现的间隔），
// 当前遗漏 = totalDraws - 12。
//
// "平均遗漏" = 期数总和 / 出现次数（即 1 / 频率，等价"周期"）。
// "最大遗漏" = 历史上最长一次未出现期数（含当前正在持续的遗漏）。

/**
 * 计算单一号码集合（reds 或 blue）下，每个号码的 freq / avgMiss / maxMiss / currentMiss。
 * @param draws 历史数据，按时间升序
 * @param size 号码空间大小（红=33，蓝=16）
 * @param field "reds" 或 "blue"
 */
export function missStats(draws, size, field = "reds") {
  const total = draws.length;
  const stats = Array.from({ length: size + 1 }, () => ({
    freq: 0,
    avgMiss: 0,
    maxMiss: 0,
    currentMiss: 0,
  }));

  for (let n = 1; n <= size; n++) {
    let lastSeen = -1; // 上次出现的索引（-1 = 从未出现）
    let maxMiss = 0;
    let freq = 0;
    for (let i = 0; i < total; i++) {
      const hit = field === "reds" ? draws[i].reds.includes(n) : draws[i].blue === n;
      if (hit) {
        const gap = i - lastSeen - 1; // 这次出现前累计未出期数
        // 首次出现前的"开局未出"也算一次遗漏（业界惯例）
        if (gap > maxMiss) maxMiss = gap;
        lastSeen = i;
        freq++;
      }
    }
    const currentMiss = lastSeen === -1 ? total : total - 1 - lastSeen;
    if (currentMiss > maxMiss) maxMiss = currentMiss;
    const avgMiss = freq === 0 ? total : total / freq - 1;
    stats[n] = { freq, avgMiss, maxMiss, currentMiss };
  }
  return stats;
}
