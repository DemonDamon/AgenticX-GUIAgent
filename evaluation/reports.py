#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Evaluation Reports
报告生成：实现评估结果的报告生成和可视化功能

Author: AgenticX Team
Date: 2025
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from loguru import logger

from utils import get_iso_timestamp
from utils import setup_logger
from .metrics import MetricResult
from .test_environment import TestResult
from .benchmarks import BenchmarkResult


class ReportFormat(Enum):
    """报告格式"""
    JSON = "json"                      # JSON格式
    HTML = "html"                      # HTML格式
    PDF = "pdf"                        # PDF格式
    CSV = "csv"                        # CSV格式
    MARKDOWN = "markdown"              # Markdown格式
    XML = "xml"                        # XML格式
    EXCEL = "excel"                    # Excel格式


class ReportType(Enum):
    """报告类型"""
    SUMMARY = "summary"                # 摘要报告
    DETAILED = "detailed"              # 详细报告
    PERFORMANCE = "performance"        # 性能报告
    COMPARISON = "comparison"          # 对比报告
    TREND = "trend"                    # 趋势报告
    BENCHMARK = "benchmark"            # 基准测试报告
    ERROR = "error"                    # 错误报告
    CUSTOM = "custom"                  # 自定义报告


class ChartType(Enum):
    """图表类型"""
    LINE = "line"                      # 折线图
    BAR = "bar"                        # 柱状图
    PIE = "pie"                        # 饼图
    SCATTER = "scatter"                # 散点图
    HISTOGRAM = "histogram"            # 直方图
    HEATMAP = "heatmap"                # 热力图
    BOX = "box"                        # 箱线图
    AREA = "area"                      # 面积图


@dataclass
class ReportConfig:
    """报告配置"""
    name: str
    title: str = ""
    description: str = ""
    report_type: ReportType = ReportType.SUMMARY
    format: ReportFormat = ReportFormat.HTML
    
    # 输出配置
    output_dir: str = "reports"
    filename: Optional[str] = None
    include_timestamp: bool = True
    
    # 内容配置
    include_summary: bool = True
    include_details: bool = True
    include_charts: bool = True
    include_raw_data: bool = False
    
    # 图表配置
    chart_types: List[ChartType] = field(default_factory=lambda: [ChartType.BAR, ChartType.LINE])
    chart_width: int = 800
    chart_height: int = 600
    
    # 样式配置
    theme: str = "default"
    color_scheme: List[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    
    # 过滤配置
    date_range: Optional[Tuple[str, str]] = None
    status_filter: Optional[List[str]] = None
    metric_filter: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'title': self.title,
            'description': self.description,
            'report_type': self.report_type.value,
            'format': self.format.value,
            'output_dir': self.output_dir,
            'filename': self.filename,
            'include_timestamp': self.include_timestamp,
            'include_summary': self.include_summary,
            'include_details': self.include_details,
            'include_charts': self.include_charts,
            'include_raw_data': self.include_raw_data,
            'chart_types': [ct.value for ct in self.chart_types],
            'chart_width': self.chart_width,
            'chart_height': self.chart_height,
            'theme': self.theme,
            'color_scheme': self.color_scheme,
            'date_range': self.date_range,
            'status_filter': self.status_filter,
            'metric_filter': self.metric_filter
        }


@dataclass
class ReportData:
    """报告数据"""
    title: str
    description: str = ""
    timestamp: str = field(default_factory=get_iso_timestamp)
    
    # 测试结果
    test_results: List[TestResult] = field(default_factory=list)
    
    # 基准测试结果
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    
    # 指标结果
    metric_results: Dict[str, MetricResult] = field(default_factory=dict)
    
    # 统计数据
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp,
            'test_results': [result.to_dict() for result in self.test_results],
            'benchmark_results': [result.to_dict() for result in self.benchmark_results],
            'metric_results': {k: v.to_dict() for k, v in self.metric_results.items()},
            'statistics': self.statistics,
            'metadata': self.metadata
        }


@dataclass
class ChartData:
    """图表数据"""
    title: str
    chart_type: ChartType
    data: Dict[str, Any]
    labels: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    width: int = 800
    height: int = 600
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'chart_type': self.chart_type.value,
            'data': self.data,
            'labels': self.labels,
            'colors': self.colors,
            'width': self.width,
            'height': self.height
        }


class BaseReportGenerator(ABC):
    """基础报告生成器"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logger
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    @abstractmethod
    async def generate(self, data: ReportData) -> str:
        """生成报告"""
        pass
    
    def _get_output_filename(self, data: ReportData) -> str:
        """获取输出文件名"""
        if self.config.filename:
            base_name = self.config.filename
        else:
            base_name = f"{self.config.name}_{data.title.replace(' ', '_')}"
        
        if self.config.include_timestamp:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp}"
        
        extension = self._get_file_extension()
        return f"{base_name}.{extension}"
    
    def _get_file_extension(self) -> str:
        """获取文件扩展名"""
        extension_map = {
            ReportFormat.JSON: "json",
            ReportFormat.HTML: "html",
            ReportFormat.PDF: "pdf",
            ReportFormat.CSV: "csv",
            ReportFormat.MARKDOWN: "md",
            ReportFormat.XML: "xml",
            ReportFormat.EXCEL: "xlsx"
        }
        return extension_map.get(self.config.format, "txt")
    
    def _filter_data(self, data: ReportData) -> ReportData:
        """过滤数据"""
        filtered_data = ReportData(
            title=data.title,
            description=data.description,
            timestamp=data.timestamp,
            metadata=data.metadata.copy()
        )
        
        # 过滤测试结果
        filtered_data.test_results = self._filter_test_results(data.test_results)
        
        # 过滤基准测试结果
        filtered_data.benchmark_results = self._filter_benchmark_results(data.benchmark_results)
        
        # 过滤指标结果
        filtered_data.metric_results = self._filter_metric_results(data.metric_results)
        
        # 重新计算统计数据
        filtered_data.statistics = self._calculate_statistics(filtered_data)
        
        return filtered_data
    
    def _filter_test_results(self, test_results: List[TestResult]) -> List[TestResult]:
        """过滤测试结果"""
        filtered = test_results
        
        # 状态过滤
        if self.config.status_filter:
            filtered = [r for r in filtered if r.status.value in self.config.status_filter]
        
        # 日期范围过滤
        if self.config.date_range:
            start_date, end_date = self.config.date_range
            filtered = [r for r in filtered if start_date <= r.start_time <= end_date]
        
        return filtered
    
    def _filter_benchmark_results(self, benchmark_results: List[BenchmarkResult]) -> List[BenchmarkResult]:
        """过滤基准测试结果"""
        filtered = benchmark_results
        
        # 状态过滤
        if self.config.status_filter:
            filtered = [r for r in filtered if r.status.value in self.config.status_filter]
        
        # 日期范围过滤
        if self.config.date_range:
            start_date, end_date = self.config.date_range
            filtered = [r for r in filtered if start_date <= r.start_time <= end_date]
        
        return filtered
    
    def _filter_metric_results(self, metric_results: Dict[str, MetricResult]) -> Dict[str, MetricResult]:
        """过滤指标结果"""
        if not self.config.metric_filter:
            return metric_results
        
        return {k: v for k, v in metric_results.items() if k in self.config.metric_filter}
    
    def _calculate_statistics(self, data: ReportData) -> Dict[str, Any]:
        """计算统计数据"""
        stats = {}
        
        # 测试结果统计
        if data.test_results:
            total_tests = len(data.test_results)
            passed_tests = sum(1 for r in data.test_results if r.status.value == "passed")
            failed_tests = sum(1 for r in data.test_results if r.status.value == "failed")
            
            stats['test_summary'] = {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'average_duration': sum(r.duration for r in data.test_results) / total_tests if total_tests > 0 else 0
            }
        
        # 基准测试结果统计
        if data.benchmark_results:
            total_benchmarks = len(data.benchmark_results)
            completed_benchmarks = sum(1 for r in data.benchmark_results if r.status.value == "completed")
            
            stats['benchmark_summary'] = {
                'total': total_benchmarks,
                'completed': completed_benchmarks,
                'completion_rate': completed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
                'total_duration': sum(r.duration for r in data.benchmark_results),
                'average_success_rate': sum(r.get_success_rate() for r in data.benchmark_results) / total_benchmarks if total_benchmarks > 0 else 0
            }
        
        # 指标统计
        if data.metric_results:
            stats['metric_summary'] = {
                'total_metrics': len(data.metric_results),
                'metrics': {k: v.value for k, v in data.metric_results.items()}
            }
        
        return stats
    
    def _generate_charts(self, data: ReportData) -> List[ChartData]:
        """生成图表数据"""
        charts = []
        
        if not self.config.include_charts:
            return charts
        
        # 测试结果图表
        if data.test_results and ChartType.PIE in self.config.chart_types:
            charts.append(self._create_test_status_pie_chart(data.test_results))
        
        if data.test_results and ChartType.BAR in self.config.chart_types:
            charts.append(self._create_test_duration_bar_chart(data.test_results))
        
        # 基准测试图表
        if data.benchmark_results and ChartType.LINE in self.config.chart_types:
            charts.append(self._create_benchmark_trend_line_chart(data.benchmark_results))
        
        # 指标图表
        if data.metric_results and ChartType.BAR in self.config.chart_types:
            charts.append(self._create_metrics_bar_chart(data.metric_results))
        
        return charts
    
    def _create_test_status_pie_chart(self, test_results: List[TestResult]) -> ChartData:
        """创建测试状态饼图"""
        status_counts = {}
        for result in test_results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return ChartData(
            title="Test Status Distribution",
            chart_type=ChartType.PIE,
            data={
                'labels': list(status_counts.keys()),
                'values': list(status_counts.values())
            },
            labels=list(status_counts.keys()),
            colors=self.config.color_scheme[:len(status_counts)],
            width=self.config.chart_width,
            height=self.config.chart_height
        )
    
    def _create_test_duration_bar_chart(self, test_results: List[TestResult]) -> ChartData:
        """创建测试持续时间柱状图"""
        test_names = [result.test_name[:20] + "..." if len(result.test_name) > 20 else result.test_name for result in test_results[:10]]
        durations = [result.duration for result in test_results[:10]]
        
        return ChartData(
            title="Test Duration (Top 10)",
            chart_type=ChartType.BAR,
            data={
                'labels': test_names,
                'values': durations
            },
            labels=test_names,
            colors=[self.config.color_scheme[0]] * len(test_names),
            width=self.config.chart_width,
            height=self.config.chart_height
        )
    
    def _create_benchmark_trend_line_chart(self, benchmark_results: List[BenchmarkResult]) -> ChartData:
        """创建基准测试趋势折线图"""
        benchmark_names = [result.benchmark_name for result in benchmark_results]
        success_rates = [result.get_success_rate() * 100 for result in benchmark_results]
        
        return ChartData(
            title="Benchmark Success Rate Trend",
            chart_type=ChartType.LINE,
            data={
                'labels': benchmark_names,
                'values': success_rates
            },
            labels=benchmark_names,
            colors=[self.config.color_scheme[1]],
            width=self.config.chart_width,
            height=self.config.chart_height
        )
    
    def _create_metrics_bar_chart(self, metric_results: Dict[str, MetricResult]) -> ChartData:
        """创建指标柱状图"""
        metric_names = list(metric_results.keys())
        metric_values = [result.value for result in metric_results.values()]
        
        return ChartData(
            title="Metrics Overview",
            chart_type=ChartType.BAR,
            data={
                'labels': metric_names,
                'values': metric_values
            },
            labels=metric_names,
            colors=self.config.color_scheme[:len(metric_names)],
            width=self.config.chart_width,
            height=self.config.chart_height
        )


class JSONReportGenerator(BaseReportGenerator):
    """JSON报告生成器"""
    
    async def generate(self, data: ReportData) -> str:
        """生成JSON报告"""
        logger.info(f"Generating JSON report: {self.config.name}")
        
        # 过滤数据
        filtered_data = self._filter_data(data)
        
        # 生成图表数据
        charts = self._generate_charts(filtered_data)
        
        # 构建报告内容
        report_content = {
            'report_info': {
                'name': self.config.name,
                'title': self.config.title or filtered_data.title,
                'description': self.config.description or filtered_data.description,
                'type': self.config.report_type.value,
                'format': self.config.format.value,
                'generated_at': get_iso_timestamp(),
                'data_timestamp': filtered_data.timestamp
            },
            'summary': filtered_data.statistics if self.config.include_summary else {},
            'data': filtered_data.to_dict() if self.config.include_details else {},
            'charts': [chart.to_dict() for chart in charts] if self.config.include_charts else [],
            'raw_data': filtered_data.to_dict() if self.config.include_raw_data else {}
        }
        
        # 写入文件
        filename = self._get_output_filename(filtered_data)
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_content, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report generated: {filepath}")
        return filepath


class HTMLReportGenerator(BaseReportGenerator):
    """HTML报告生成器"""
    
    async def generate(self, data: ReportData) -> str:
        """生成HTML报告"""
        logger.info(f"Generating HTML report: {self.config.name}")
        
        # 过滤数据
        filtered_data = self._filter_data(data)
        
        # 生成图表数据
        charts = self._generate_charts(filtered_data)
        
        # 构建HTML内容
        html_content = self._build_html_content(filtered_data, charts)
        
        # 写入文件
        filename = self._get_output_filename(filtered_data)
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filepath}")
        return filepath
    
    def _build_html_content(self, data: ReportData, charts: List[ChartData]) -> str:
        """构建HTML内容"""
        title = self.config.title or data.title
        description = self.config.description or data.description
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="description">{description}</p>
            <p class="timestamp">Generated at: {get_iso_timestamp()}</p>
        </header>
        
        {self._build_summary_section(data) if self.config.include_summary else ''}
        
        {self._build_charts_section(charts) if self.config.include_charts else ''}
        
        {self._build_details_section(data) if self.config.include_details else ''}
    </div>
    
    {self._build_chart_scripts(charts) if self.config.include_charts else ''}
</body>
</html>
"""
        return html
    
    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .description {
            font-size: 16px;
            color: #666;
            margin-bottom: 10px;
        }
        
        .timestamp {
            font-size: 14px;
            color: #999;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section h2 {
            color: #34495e;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .summary-card h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #27ae60;
        }
        
        .chart-container {
            margin-bottom: 30px;
            text-align: center;
        }
        
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        canvas {
            max-width: 100%;
            height: auto;
        }
        
        .details-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .details-table th,
        .details-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .details-table th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .status-passed {
            color: #27ae60;
            font-weight: bold;
        }
        
        .status-failed {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .status-skipped {
            color: #f39c12;
            font-weight: bold;
        }
        """
    
    def _build_summary_section(self, data: ReportData) -> str:
        """构建摘要部分"""
        if not data.statistics:
            return ""
        
        summary_cards = []
        
        # 测试摘要卡片
        if 'test_summary' in data.statistics:
            test_stats = data.statistics['test_summary']
            summary_cards.append(f"""
            <div class="summary-card">
                <h3>Test Summary</h3>
                <div class="metric">
                    <span>Total Tests:</span>
                    <span class="metric-value">{test_stats['total']}</span>
                </div>
                <div class="metric">
                    <span>Passed:</span>
                    <span class="metric-value">{test_stats['passed']}</span>
                </div>
                <div class="metric">
                    <span>Failed:</span>
                    <span class="metric-value">{test_stats['failed']}</span>
                </div>
                <div class="metric">
                    <span>Success Rate:</span>
                    <span class="metric-value">{test_stats['success_rate']:.2%}</span>
                </div>
                <div class="metric">
                    <span>Avg Duration:</span>
                    <span class="metric-value">{test_stats['average_duration']:.2f}s</span>
                </div>
            </div>
            """)
        
        # 基准测试摘要卡片
        if 'benchmark_summary' in data.statistics:
            bench_stats = data.statistics['benchmark_summary']
            summary_cards.append(f"""
            <div class="summary-card">
                <h3>Benchmark Summary</h3>
                <div class="metric">
                    <span>Total Benchmarks:</span>
                    <span class="metric-value">{bench_stats['total']}</span>
                </div>
                <div class="metric">
                    <span>Completed:</span>
                    <span class="metric-value">{bench_stats['completed']}</span>
                </div>
                <div class="metric">
                    <span>Completion Rate:</span>
                    <span class="metric-value">{bench_stats['completion_rate']:.2%}</span>
                </div>
                <div class="metric">
                    <span>Total Duration:</span>
                    <span class="metric-value">{bench_stats['total_duration']:.2f}s</span>
                </div>
                <div class="metric">
                    <span>Avg Success Rate:</span>
                    <span class="metric-value">{bench_stats['average_success_rate']:.2%}</span>
                </div>
            </div>
            """)
        
        # 指标摘要卡片
        if 'metric_summary' in data.statistics:
            metric_stats = data.statistics['metric_summary']
            metrics_html = ""
            for metric_name, metric_value in metric_stats['metrics'].items():
                metrics_html += f"""
                <div class="metric">
                    <span>{metric_name}:</span>
                    <span class="metric-value">{metric_value:.2f}</span>
                </div>
                """
            
            summary_cards.append(f"""
            <div class="summary-card">
                <h3>Metrics Summary</h3>
                <div class="metric">
                    <span>Total Metrics:</span>
                    <span class="metric-value">{metric_stats['total_metrics']}</span>
                </div>
                {metrics_html}
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>Summary</h2>
            <div class="summary-grid">
                {''.join(summary_cards)}
            </div>
        </div>
        """
    
    def _build_charts_section(self, charts: List[ChartData]) -> str:
        """构建图表部分"""
        if not charts:
            return ""
        
        charts_html = []
        for i, chart in enumerate(charts):
            charts_html.append(f"""
            <div class="chart-container">
                <div class="chart-title">{chart.title}</div>
                <canvas id="chart_{i}" width="{chart.width}" height="{chart.height}"></canvas>
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>Charts</h2>
            {''.join(charts_html)}
        </div>
        """
    
    def _build_details_section(self, data: ReportData) -> str:
        """构建详情部分"""
        details_html = []
        
        # 测试结果详情
        if data.test_results:
            test_rows = []
            for result in data.test_results:
                status_class = f"status-{result.status.value}"
                test_rows.append(f"""
                <tr>
                    <td>{result.test_name}</td>
                    <td class="{status_class}">{result.status.value.upper()}</td>
                    <td>{result.duration:.2f}s</td>
                    <td>{result.start_time}</td>
                    <td>{result.error_message or '-'}</td>
                </tr>
                """)
            
            details_html.append(f"""
            <h3>Test Results</h3>
            <table class="details-table">
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Start Time</th>
                        <th>Error Message</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(test_rows)}
                </tbody>
            </table>
            """)
        
        # 基准测试结果详情
        if data.benchmark_results:
            benchmark_rows = []
            for result in data.benchmark_results:
                status_class = f"status-{result.status.value}"
                benchmark_rows.append(f"""
                <tr>
                    <td>{result.benchmark_name}</td>
                    <td class="{status_class}">{result.status.value.upper()}</td>
                    <td>{result.duration:.2f}s</td>
                    <td>{result.iterations_completed}/{result.total_iterations}</td>
                    <td>{result.get_success_rate():.2%}</td>
                    <td>{result.error_message or '-'}</td>
                </tr>
                """)
            
            details_html.append(f"""
            <h3>Benchmark Results</h3>
            <table class="details-table">
                <thead>
                    <tr>
                        <th>Benchmark Name</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Iterations</th>
                        <th>Success Rate</th>
                        <th>Error Message</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(benchmark_rows)}
                </tbody>
            </table>
            """)
        
        if not details_html:
            return ""
        
        return f"""
        <div class="section">
            <h2>Details</h2>
            {''.join(details_html)}
        </div>
        """
    
    def _build_chart_scripts(self, charts: List[ChartData]) -> str:
        """构建图表脚本"""
        if not charts:
            return ""
        
        scripts = []
        for i, chart in enumerate(charts):
            script = self._generate_chart_script(i, chart)
            scripts.append(script)
        
        return f"""
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                {''.join(scripts)}
            }});
        </script>
        """
    
    def _generate_chart_script(self, chart_id: int, chart: ChartData) -> str:
        """生成图表脚本"""
        chart_type_map = {
            ChartType.PIE: "pie",
            ChartType.BAR: "bar",
            ChartType.LINE: "line",
            ChartType.SCATTER: "scatter",
            ChartType.AREA: "line"
        }
        
        chart_type = chart_type_map.get(chart.chart_type, "bar")
        
        if chart.chart_type == ChartType.PIE:
            return f"""
                var ctx_{chart_id} = document.getElementById('chart_{chart_id}').getContext('2d');
                new Chart(ctx_{chart_id}, {{
                    type: '{chart_type}',
                    data: {{
                        labels: {json.dumps(chart.data.get('labels', []))},
                        datasets: [{{
                            data: {json.dumps(chart.data.get('values', []))},
                            backgroundColor: {json.dumps(chart.colors)}
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}
                }});
            """
        else:
            return f"""
                var ctx_{chart_id} = document.getElementById('chart_{chart_id}').getContext('2d');
                new Chart(ctx_{chart_id}, {{
                    type: '{chart_type}',
                    data: {{
                        labels: {json.dumps(chart.data.get('labels', []))},
                        datasets: [{{
                            label: '{chart.title}',
                            data: {json.dumps(chart.data.get('values', []))},
                            backgroundColor: '{chart.colors[0] if chart.colors else "#1f77b4"}',
                            borderColor: '{chart.colors[0] if chart.colors else "#1f77b4"}',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});
            """


class MarkdownReportGenerator(BaseReportGenerator):
    """Markdown报告生成器"""
    
    async def generate(self, data: ReportData) -> str:
        """生成Markdown报告"""
        logger.info(f"Generating Markdown report: {self.config.name}")
        
        # 过滤数据
        filtered_data = self._filter_data(data)
        
        # 构建Markdown内容
        markdown_content = self._build_markdown_content(filtered_data)
        
        # 写入文件
        filename = self._get_output_filename(filtered_data)
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report generated: {filepath}")
        return filepath
    
    def _build_markdown_content(self, data: ReportData) -> str:
        """构建Markdown内容"""
        title = self.config.title or data.title
        description = self.config.description or data.description
        
        content = f"""# {title}

{description}

**Generated at:** {get_iso_timestamp()}  
**Data timestamp:** {data.timestamp}

---

"""
        
        # 摘要部分
        if self.config.include_summary and data.statistics:
            content += self._build_markdown_summary(data.statistics)
        
        # 详情部分
        if self.config.include_details:
            content += self._build_markdown_details(data)
        
        return content
    
    def _build_markdown_summary(self, statistics: Dict[str, Any]) -> str:
        """构建Markdown摘要"""
        content = "## Summary\n\n"
        
        # 测试摘要
        if 'test_summary' in statistics:
            test_stats = statistics['test_summary']
            content += f"""### Test Summary

| Metric | Value |
|--------|-------|
| Total Tests | {test_stats['total']} |
| Passed | {test_stats['passed']} |
| Failed | {test_stats['failed']} |
| Success Rate | {test_stats['success_rate']:.2%} |
| Average Duration | {test_stats['average_duration']:.2f}s |

"""
        
        # 基准测试摘要
        if 'benchmark_summary' in statistics:
            bench_stats = statistics['benchmark_summary']
            content += f"""### Benchmark Summary

| Metric | Value |
|--------|-------|
| Total Benchmarks | {bench_stats['total']} |
| Completed | {bench_stats['completed']} |
| Completion Rate | {bench_stats['completion_rate']:.2%} |
| Total Duration | {bench_stats['total_duration']:.2f}s |
| Average Success Rate | {bench_stats['average_success_rate']:.2%} |

"""
        
        # 指标摘要
        if 'metric_summary' in statistics:
            metric_stats = statistics['metric_summary']
            content += f"""### Metrics Summary

| Metric | Value |
|--------|-------|
| Total Metrics | {metric_stats['total_metrics']} |
"""
            
            for metric_name, metric_value in metric_stats['metrics'].items():
                content += f"| {metric_name} | {metric_value:.2f} |\n"
            
            content += "\n"
        
        return content
    
    def _build_markdown_details(self, data: ReportData) -> str:
        """构建Markdown详情"""
        content = "## Details\n\n"
        
        # 测试结果详情
        if data.test_results:
            content += """### Test Results

| Test Name | Status | Duration | Start Time | Error Message |
|-----------|--------|----------|------------|---------------|
"""
            
            for result in data.test_results:
                error_msg = result.error_message or '-'
                content += f"| {result.test_name} | {result.status.value.upper()} | {result.duration:.2f}s | {result.start_time} | {error_msg} |\n"
            
            content += "\n"
        
        # 基准测试结果详情
        if data.benchmark_results:
            content += """### Benchmark Results

| Benchmark Name | Status | Duration | Iterations | Success Rate | Error Message |
|----------------|--------|----------|------------|--------------|---------------|
"""
            
            for result in data.benchmark_results:
                error_msg = result.error_message or '-'
                content += f"| {result.benchmark_name} | {result.status.value.upper()} | {result.duration:.2f}s | {result.iterations_completed}/{result.total_iterations} | {result.get_success_rate():.2%} | {error_msg} |\n"
            
            content += "\n"
        
        return content


class ReportManager:
    """报告管理器"""
    
    def __init__(self):
        self.logger = logger
        
        # 注册的生成器
        self.generators: Dict[ReportFormat, type] = {
            ReportFormat.JSON: JSONReportGenerator,
            ReportFormat.HTML: HTMLReportGenerator,
            ReportFormat.MARKDOWN: MarkdownReportGenerator
        }
        
        # 生成历史
        self.generation_history: List[Dict[str, Any]] = []
    
    def register_generator(self, format: ReportFormat, generator_class: type) -> None:
        """注册报告生成器"""
        self.generators[format] = generator_class
        logger.info(f"Registered generator for format: {format.value}")
    
    async def generate_report(self, config: ReportConfig, data: ReportData) -> Optional[str]:
        """生成报告"""
        generator_class = self.generators.get(config.format)
        if not generator_class:
            logger.error(f"No generator found for format: {config.format.value}")
            return None
        
        try:
            generator = generator_class(config)
            filepath = await generator.generate(data)
            
            # 记录生成历史
            self.generation_history.append({
                'config_name': config.name,
                'format': config.format.value,
                'filepath': filepath,
                'timestamp': get_iso_timestamp(),
                'data_title': data.title
            })
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate report {config.name}: {e}")
            return None
    
    async def generate_multiple_reports(self, configs: List[ReportConfig], data: ReportData) -> List[str]:
        """生成多个报告"""
        filepaths = []
        
        for config in configs:
            filepath = await self.generate_report(config, data)
            if filepath:
                filepaths.append(filepath)
        
        return filepaths
    
    def get_supported_formats(self) -> List[ReportFormat]:
        """获取支持的格式"""
        return list(self.generators.keys())
    
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """获取生成历史"""
        return self.generation_history.copy()
    
    def clear_history(self) -> None:
        """清空生成历史"""
        self.generation_history.clear()
        logger.info("Cleared generation history")