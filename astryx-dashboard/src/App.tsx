// Astryx (Meta design system) front-end demo for the imbalance price forecast.
// Mirrors the Streamlit situational view in ../app.py, reading the JSON
// snapshot exported by scripts/export_data.py. Demo only - not production.

import {useEffect, useMemo, useState} from 'react';
import {VStack, HStack, Layout, LayoutContent} from '@astryxdesign/core/Layout';
import {Text, Heading} from '@astryxdesign/core/Text';
import {Card} from '@astryxdesign/core/Card';
import {Badge} from '@astryxdesign/core/Badge';
import {Banner} from '@astryxdesign/core/Banner';
import {TabList, Tab} from '@astryxdesign/core/TabList';
import {Selector} from '@astryxdesign/core/Selector';
import {Table, pixel, proportional} from '@astryxdesign/core/Table';
import type {TableColumn} from '@astryxdesign/core/Table';
import {Grid} from '@astryxdesign/core/Grid';
import {Spinner} from '@astryxdesign/core/Spinner';
import {Icon} from '@astryxdesign/core/Icon';
import {
  ComposedChart,
  Area,
  Line,
  BarChart,
  Bar,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {StopIcon} from '@heroicons/react/24/solid';

// ------------------------------------------------------------------ types

interface ForecastRow {
  step: number;
  t: string;
  p10: number | null;
  p50: number | null;
  p90: number | null;
  baseline: number | null;
  actual: number | null;
  spot: number | null;
  pUp: number | null;
  pDown: number | null;
  pNone: number | null;
  pSpikeHigh: number | null;
  pSpikeLow: number | null;
}

interface OriginData {
  origin: string;
  label: string;
  dataCutoff: string;
  topDrivers: string;
  rows: ForecastRow[];
}

interface MetricRow extends Record<string, unknown> {
  id: string;
  model: string;
  MAE_p50_eur_mwh: number | null;
  pinball_mean: number | null;
  coverage_p10_p90: number | null;
  brier_spike_high: number | null;
  brier_spike_low: number | null;
  brier_direction: number | null;
}

interface DemoData {
  generatedFrom: 'synthetic' | 'real';
  originCount: number;
  origins: OriginData[];
  metrics: MetricRow[];
  maeByOrigin: {origin: string; model: number | null; baseline: number | null}[];
  topDrivers: string;
}

// ------------------------------------------------------------------ colors

const colors = {
  band: 'rgba(1, 113, 227, 0.15)',
  p50: 'var(--color-data-categorical-blue, #0171E3)',
  baseline: 'var(--color-data-neutral, #8494A3)',
  actual: 'var(--color-data-categorical-green, #0B991F)',
  spot: 'var(--color-data-categorical-orange, #EB6E00)',
  up: 'var(--color-data-categorical-red, #E32B2B)',
  down: 'var(--color-data-categorical-blue, #0171E3)',
  none: 'var(--color-data-neutral, #8494A3)',
};

function LegendItem({color, label, dash}: {color: string; label: string; dash?: boolean}) {
  return (
    <HStack gap={2} vAlign="center">
      <Icon icon={StopIcon} size="xsm" style={{color}} />
      <Text type="supporting" color="secondary">
        {label}
        {dash ? ' (dashed)' : ''}
      </Text>
    </HStack>
  );
}

const tickStyle = {
  fontSize: 'var(--font-size-sm, 12px)',
  fill: 'var(--color-text-secondary, #4E606F)',
} as const;

// ------------------------------------------------------------------ charts

interface ChartPoint {
  t: string;
  step: number;
  band: [number, number] | null;
  p50: number | null;
  baseline: number | null;
  actual: number | null;
  spot: number | null;
}

function toPoints(rows: ForecastRow[]): ChartPoint[] {
  return rows.map(r => ({
    t: r.t,
    step: r.step,
    band: r.p10 != null && r.p90 != null ? [r.p10, r.p90] : null,
    p50: r.p50,
    baseline: r.baseline,
    actual: r.actual,
    spot: r.spot,
  }));
}

function ForecastChart({rows, height = 380}: {rows: ForecastRow[]; height?: number}) {
  const data = toPoints(rows);
  const every8 = data.filter((_, i) => i % 8 === 0).map(d => d.t);
  return (
    <VStack gap={3}>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{top: 5, right: 10, left: 0, bottom: 5}}>
          <CartesianGrid horizontal vertical={false} stroke="var(--color-border, rgba(5,54,89,0.1))" />
          <XAxis dataKey="t" ticks={every8} tick={tickStyle} axisLine={false} tickLine={false} />
          <YAxis
            tick={tickStyle}
            axisLine={false}
            tickLine={false}
            width={48}
            label={{value: 'EUR/MWh', angle: -90, position: 'insideLeft', style: {fontSize: 12, fill: '#4E606F'}}}
          />
          <Tooltip />
          <Area dataKey="band" name="P10-P90 interval" fill={colors.band} stroke="none" connectNulls />
          <Line type="linear" dataKey="p50" name="P50 forecast" stroke={colors.p50} strokeWidth={2} dot={false} connectNulls />
          <Line type="linear" dataKey="baseline" name="Baseline (last price)" stroke={colors.baseline} strokeDasharray="3 3" dot={false} connectNulls />
          <Line type="linear" dataKey="actual" name="Actual (known later)" stroke={colors.actual} strokeDasharray="6 3" dot={false} connectNulls />
          <Line type="linear" dataKey="spot" name="Spot" stroke={colors.spot} strokeDasharray="8 4 2 4" dot={false} connectNulls />
        </ComposedChart>
      </ResponsiveContainer>
      <HStack gap={6} wrap="wrap">
        <LegendItem color={colors.p50} label="P50 forecast" />
        <LegendItem color={colors.baseline} label="Baseline" />
        <LegendItem color={colors.actual} label="Actual" dash />
        <LegendItem color={colors.spot} label="Spot" dash />
      </HStack>
    </VStack>
  );
}

function DirectionChart({rows}: {rows: ForecastRow[]}) {
  const data = rows.map(r => ({t: r.t, Up: r.pUp ?? 0, Down: r.pDown ?? 0, None: r.pNone ?? 0}));
  const every8 = data.filter((_, i) => i % 8 === 0).map(d => d.t);
  return (
    <VStack gap={3}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{top: 5, right: 10, left: 0, bottom: 5}}>
          <CartesianGrid horizontal vertical={false} stroke="var(--color-border, rgba(5,54,89,0.1))" />
          <XAxis dataKey="t" ticks={every8} tick={tickStyle} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={tickStyle} axisLine={false} tickLine={false} width={40} />
          <Tooltip />
          <Bar dataKey="Up" stackId="d" fill={colors.up} isAnimationActive={false} />
          <Bar dataKey="Down" stackId="d" fill={colors.down} isAnimationActive={false} />
          <Bar dataKey="None" stackId="d" fill={colors.none} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <HStack gap={6}>
        <LegendItem color={colors.up} label="Up-regulation" />
        <LegendItem color={colors.down} label="Down-regulation" />
        <LegendItem color={colors.none} label="No dominating direction" />
      </HStack>
    </VStack>
  );
}

// ------------------------------------------------------------------ pieces

function MetricCard({label, value, note}: {label: string; value: string; note?: string}) {
  return (
    <Card>
      <VStack gap={2}>
        <Heading level={4}>{label}</Heading>
        <Heading level={2}>{value}</Heading>
        {note ? (
          <Text type="supporting" color="secondary">
            {note}
          </Text>
        ) : null}
      </VStack>
    </Card>
  );
}

function OriginSelector({
  origins,
  value,
  onChange,
}: {
  origins: OriginData[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <Selector
      label="Forecast origin (move the viewing time into the past)"
      options={origins.map(o => ({value: o.origin, label: o.label}))}
      value={value}
      onChange={onChange}
      width={360}
    />
  );
}

function pct(v: number | null): string {
  return v == null ? 'n/a' : `${Math.round(v * 100)}%`;
}

// ------------------------------------------------------------------ tabs

function NextTenHours({data}: {data: DemoData}) {
  const [origin, setOrigin] = useState(data.origins[data.origins.length - 1].origin);
  const od = data.origins.find(o => o.origin === origin)!;
  const rows = od.rows;
  const maxHigh = Math.max(...rows.map(r => r.pSpikeHigh ?? 0));
  const maxLow = Math.max(...rows.map(r => r.pSpikeLow ?? 0));
  const widths = rows.filter(r => r.p10 != null && r.p90 != null).map(r => r.p90! - r.p10!);
  const meanWidth = widths.length ? widths.reduce((a, b) => a + b, 0) / widths.length : null;
  const hasSpikes = rows.some(r => r.pSpikeHigh != null);

  return (
    <VStack gap={6}>
      <OriginSelector origins={data.origins} value={origin} onChange={setOrigin} />
      <Card>
        <VStack gap={3}>
          <Heading level={3}>Forecast from {od.label}</Heading>
          <ForecastChart rows={rows} />
        </VStack>
      </Card>
      <Grid columns={{minWidth: 240, repeat: 'fit'}} gap={4}>
        <MetricCard
          label="Highest spike risk (> spot + 50)"
          value={hasSpikes ? pct(maxHigh) : 'n/a'}
        />
        <MetricCard
          label="Highest low-price risk (< spot - 50)"
          value={hasSpikes ? pct(maxLow) : 'n/a'}
        />
        <MetricCard
          label="Mean P10-P90 width"
          value={meanWidth == null ? 'n/a' : `${meanWidth.toFixed(0)} EUR/MWh`}
        />
      </Grid>
      {!hasSpikes ? (
        <Text type="supporting" color="secondary">
          Exceptional-price probabilities need the Nord Pool spot price, which is not open data -
          not supplied for this run, so they are empty on purpose.
        </Text>
      ) : null}
      <Card>
        <VStack gap={3}>
          <Heading level={3}>Direction of the balancing need</Heading>
          <DirectionChart rows={rows} />
          <Text type="supporting" color="secondary">
            Data cutoff: {od.dataCutoff} - main drivers (global): {od.topDrivers}
          </Text>
        </VStack>
      </Card>
    </VStack>
  );
}

function Alerts({data}: {data: DemoData}) {
  const [origin, setOrigin] = useState(data.origins[data.origins.length - 1].origin);
  const od = data.origins.find(o => o.origin === origin)!;
  const rows = od.rows;
  const hasSpikes = rows.some(r => r.pSpikeHigh != null);
  const hi = hasSpikes ? rows.filter(r => (r.pSpikeHigh ?? 0) >= 0.15) : [];
  const lo = hasSpikes ? rows.filter(r => (r.pSpikeLow ?? 0) >= 0.15) : [];
  const first = rows[0];
  const dirOptions: [string, number][] = [
    ['up-regulation', first.pUp ?? 0],
    ['down-regulation', first.pDown ?? 0],
    ['no dominating direction', first.pNone ?? 0],
  ];
  const dir = dirOptions.sort((a, b) => b[1] - a[1])[0];

  return (
    <VStack gap={6}>
      <OriginSelector origins={data.origins} value={origin} onChange={setOrigin} />
      <Heading level={3}>Situation at {od.label}</Heading>
      {hi.length ? (
        <Banner
          status="error"
          title={`Demo high-price risk exceeds the threshold for ${hi[0].t}-${hi[hi.length - 1].t} UTC`}
          description={`Global model associations: ${od.topDrivers} (not a local explanation). Uncertainty is shown by the P10-P90 band; if it is wide, the production plan and market position should be checked.`}
        />
      ) : null}
      {lo.length ? (
        <Banner
          status="warning"
          title={`Demo low-price risk exceeds the threshold for ${lo[0].t}-${lo[lo.length - 1].t} UTC`}
          description={`Global model associations: ${od.topDrivers} (not a local explanation). Worth checking whether flexible consumption or production can be shifted into the window.`}
        />
      ) : null}
      {!hi.length && !lo.length ? (
        <Banner
          status="success"
          title="No spike alerts at the demo threshold in the next 10 hours"
          description={`Most likely direction: ${dir[0]} (${pct(dir[1])}).`}
        />
      ) : null}
      <Banner
        status="info"
        title="Suggestion logic (demo)"
        description="Alert when a spike probability reaches 15% anywhere in the horizon, then attach the model's main global drivers and an action hint. Thresholds and wording are placeholders to tune with Turku Energia's experts."
      />
    </VStack>
  );
}

const metricColumns: TableColumn<MetricRow>[] = [
  {key: 'model', header: 'Model', width: proportional(1)},
  {
    key: 'MAE_p50_eur_mwh',
    header: 'P50 MAE (EUR/MWh)',
    width: pixel(150),
    renderCell: m => m.MAE_p50_eur_mwh?.toFixed(3) ?? 'n/a',
  },
  {
    key: 'pinball_mean',
    header: 'Mean pinball P10/P90',
    width: pixel(160),
    renderCell: m => m.pinball_mean?.toFixed(3) ?? 'n/a',
  },
  {
    key: 'coverage_p10_p90',
    header: 'P10-P90 coverage',
    width: pixel(140),
    renderCell: m => (m.coverage_p10_p90 == null ? 'n/a' : `${(m.coverage_p10_p90 * 100).toFixed(1)}%`),
  },
  {
    key: 'brier_spike_high',
    header: 'Brier high spike',
    width: pixel(130),
    renderCell: m => m.brier_spike_high?.toFixed(4) ?? 'n/a',
  },
  {
    key: 'brier_spike_low',
    header: 'Brier low spike',
    width: pixel(130),
    renderCell: m => m.brier_spike_low?.toFixed(4) ?? 'n/a',
  },
  {
    key: 'brier_direction',
    header: 'Brier direction',
    width: pixel(130),
    renderCell: m => m.brier_direction?.toFixed(4) ?? 'n/a',
  },
];

function PastPerformance({data}: {data: DemoData}) {
  const [origin, setOrigin] = useState(data.origins[data.origins.length - 1].origin);
  const od = data.origins.find(o => o.origin === origin)!;
  const maeData = data.maeByOrigin;
  const ticks = maeData.filter((_, i) => i % Math.ceil(maeData.length / 6) === 0).map(d => d.origin);
  return (
    <VStack gap={6}>
      <Heading level={3}>Retrospective benchmark: 5 Oct - 1 Nov 2026</Heading>
      <Card>
        <Table<MetricRow> data={data.metrics} columns={metricColumns} idKey="id" density="compact" dividers="rows" hasHover />
      </Card>
      <Text type="supporting" color="secondary">
        Coverage target is about 0.80.{' '}
        {data.generatedFrom === 'real'
          ? 'Real Fingrid open data, time-based backtest. The ML point forecast does not beat the spot-only baseline on the October retrospective benchmark.'
          : 'All figures on synthetic sample data - they validate the pipeline, not real skill.'}
      </Text>
      <Card>
        <VStack gap={3}>
          <Heading level={4}>MAE per forecast origin</Heading>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={maeData} margin={{top: 5, right: 10, left: 0, bottom: 5}}>
              <CartesianGrid horizontal vertical={false} stroke="var(--color-border, rgba(5,54,89,0.1))" />
              <XAxis dataKey="origin" ticks={ticks} tick={tickStyle} axisLine={false} tickLine={false} />
              <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={40} />
              <Tooltip />
              <Line type="linear" dataKey="model" name="Model MAE / origin" stroke={colors.p50} strokeWidth={1.5} dot={false} />
              <Line type="linear" dataKey="baseline" name="Baseline MAE / origin" stroke={colors.baseline} strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <HStack gap={6}>
            <LegendItem color={colors.p50} label="Model MAE / origin" />
            <LegendItem color={colors.baseline} label="Baseline MAE / origin" />
          </HStack>
        </VStack>
      </Card>
      <OriginSelector origins={data.origins} value={origin} onChange={setOrigin} />
      <Card>
        <VStack gap={3}>
          <Heading level={4}>Replay: {od.label}</Heading>
          <ForecastChart rows={od.rows} height={320} />
        </VStack>
      </Card>
    </VStack>
  );
}

function About() {
  return (
    <VStack gap={4}>
      <Heading level={3}>About this demo</Heading>
      <Text type="body">
        This is an Astryx (Meta's open-source design system, React 19) front-end for the
        10-hour imbalance price forecast research demo by Toki Mohammad Tahmid. It mirrors the
        Streamlit situational view in this repository (app.py): P50 forecast with P10-P90
        interval, direction of the balancing need, exceptional-price probabilities, and the
        retrospective benchmark.
      </Text>
      <Text type="body">
        The numbers come from the same backtest pipeline (backtest.py) exported to JSON by
        scripts/export_data.py. Forecasting, metrics and alert logic are unchanged; this app
        only re-presents them. It is a demo, not a production tool: no claim of future
        accuracy, production readiness or profitability is made.
      </Text>
      <Text type="body">
        Run it: npm install, then npm run dev. Regenerate the data snapshot with
        python backtest.py and then python astryx-dashboard/scripts/export_data.py from the
        repo root.
      </Text>
    </VStack>
  );
}

// ------------------------------------------------------------------ app

export default function App() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState('now');

  useEffect(() => {
    fetch('data/demo-data.json')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: DemoData) => {
        d.metrics = d.metrics.map((m, i) => ({...m, id: String(i)}));
        setData(d);
      })
      .catch(e => setError(String(e)));
  }, []);

  const synthetic = data?.generatedFrom !== 'real';

  return (
    <Layout
      height="fill"
      content={
        <LayoutContent padding={6}>
          <VStack gap={6}>
            <HStack gap={3} vAlign="center">
              <Heading level={1}>Imbalance price forecast - situational view</Heading>
              <Badge variant="purple" label="Forecast dashboard" />
            </HStack>
            <Text type="supporting" color="secondary">
              Draft for Turku Energia's imbalance price challenge, Toki Mohammad Tahmid. Forecast:
              P50 with P10-P90 interval, 40 quarter-hours ahead from each full-hour origin.
            </Text>
            {data ? (
              synthetic ? (
                <Banner
                  status="warning"
                  title="Data source: SYNTHETIC SAMPLE DATA"
                  description="Generated numbers that imitate the shape of the real series. Metrics measure the code on that data, not real forecasting skill."
                />
              ) : (
                <Banner
                  status="warning"
                  title="Data source: REAL FINGRID OPEN DATA (CC-BY)"
                  description="A retrospective benchmark on historical data with availability assumptions. Not a live forecast. The nominal 80% band covered about 90% in October, so it was too wide."
                />
              )
            ) : null}
            {error ? (
              <Banner status="error" title="Could not load demo data" description={error} />
            ) : null}
            {!data && !error ? <Spinner label="Loading demo data" /> : null}
            {data ? (
              <>
                <TabList value={tab} onChange={setTab} hasDivider>
                  <Tab value="now" label="Next 10 hours" />
                  <Tab value="alerts" label="What to check now" />
                  <Tab value="past" label="Past performance" />
                  <Tab value="about" label="About" />
                </TabList>
                {tab === 'now' ? <NextTenHours data={data} /> : null}
                {tab === 'alerts' ? <Alerts data={data} /> : null}
                {tab === 'past' ? <PastPerformance data={data} /> : null}
                {tab === 'about' ? <About /> : null}
              </>
            ) : null}
          </VStack>
        </LayoutContent>
      }
    />
  );
      }
