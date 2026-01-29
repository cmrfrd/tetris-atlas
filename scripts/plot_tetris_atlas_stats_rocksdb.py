#!/usr/bin/env python3
"""
Visualize Tetris Atlas RocksDB Statistics

This script reads the tetris_atlas_stats_rocksdb.csv file and creates comprehensive
visualizations of all the key metrics including:
- Atlas expansion (boards, lookup table, frontier queue)
- Performance rates (boards/sec, frontier consumption/expansion)
- Cache performance
- Game outcomes
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def human_readable_size(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_value(val):
    """Format value for display in annotations"""
    if val >= 1e9:
        return f"{val/1e9:.2f}B"
    elif val >= 1e6:
        return f"{val/1e6:.2f}M"
    elif val >= 1e3:
        return f"{val/1e3:.2f}K"
    else:
        return f"{val:.2f}"


def add_current_value_annotation(ax, x_data, y_data, color, label=None, suffix=""):
    """Add a bubble showing the current (final) value on a plot"""
    if len(x_data) == 0 or len(y_data) == 0:
        return

    final_x = x_data.iloc[-1]
    final_y = y_data.iloc[-1]

    # Format the value
    if suffix == "%":
        text = f"{final_y:.1f}{suffix}"
    else:
        text = format_value(final_y) + suffix

    # Add annotation with bubble
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.annotate(text, xy=(final_x, final_y), xytext=(10, 0), textcoords='offset points',
                fontsize=9, color='white', fontweight='bold', bbox=bbox_props,
                ha='left', va='center')


def plot_tetris_atlas_stats_rocksdb(csv_path, output_path=None):
    """Create comprehensive visualization of Tetris Atlas RocksDB statistics"""

    # Read CSV
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} data points")
    print(f"Time range: {df['timestamp_secs'].min():.1f}s to {df['timestamp_secs'].max():.1f}s")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    time = df['timestamp_secs']

    # ==================== Row 1: Atlas Expansion ====================

    # 1. Atlas Size Growth
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, df['lookup_size'], label='Lookup Table', linewidth=2, color='#2E86AB')
    ax1.plot(time, df['frontier_size'], label='Frontier Queue', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Time (seconds)', fontsize=10)
    ax1.set_ylabel('Number of Entries', fontsize=10)
    ax1.set_title('Atlas Size Growth', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    add_current_value_annotation(ax1, time, df['lookup_size'], '#2E86AB')
    add_current_value_annotation(ax1, time, df['frontier_size'], '#A23B72')

    # 2. Boards Expanded
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, df['boards_expanded'], linewidth=2, color='#F18F01')
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('Boards Expanded', fontsize=10)
    ax2.set_title('Total Boards Expanded', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    add_current_value_annotation(ax2, time, df['boards_expanded'], '#F18F01')

    # 3. Frontier Operations
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time, df['frontier_enqueued'], label='Enqueued', linewidth=2, color='#06A77D')
    ax3.plot(time, df['frontier_consumed'], label='Consumed', linewidth=2, color='#A23B72')
    ax3.plot(time, df['frontier_deleted'], label='Deleted', linewidth=2, color='#D62246', linestyle='--')
    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Frontier Queue Operations', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    add_current_value_annotation(ax3, time, df['frontier_enqueued'], '#06A77D')
    add_current_value_annotation(ax3, time, df['frontier_consumed'], '#A23B72')
    add_current_value_annotation(ax3, time, df['frontier_deleted'], '#D62246')

    # ==================== Row 2: Performance Rates ====================

    # 4. Boards Per Second
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time, df['boards_per_sec'], linewidth=2, color='#F18F01')
    ax4.set_xlabel('Time (seconds)', fontsize=10)
    ax4.set_ylabel('Boards/sec', fontsize=10)
    ax4.set_title('Board Expansion Rate', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    add_current_value_annotation(ax4, time, df['boards_per_sec'], '#F18F01', suffix='/s')

    # 5. Frontier Consumption & Expansion Rates
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(time, df['frontier_consumption_rate'], label='Consumption', linewidth=2, color='#A23B72')
    ax5.plot(time, df['frontier_expansion_rate'], label='Expansion', linewidth=2, color='#06A77D')
    ax5.set_xlabel('Time (seconds)', fontsize=10)
    ax5.set_ylabel('Items/sec', fontsize=10)
    ax5.set_title('Frontier Queue Rates', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    add_current_value_annotation(ax5, time, df['frontier_consumption_rate'], '#A23B72', suffix='/s')
    add_current_value_annotation(ax5, time, df['frontier_expansion_rate'], '#06A77D', suffix='/s')

    # 6. Frontier Expansion Ratio
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(time, df['frontier_expansion_ratio'], linewidth=2, color='#7209B7')
    ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ratio = 1.0')
    ax6.set_xlabel('Time (seconds)', fontsize=10)
    ax6.set_ylabel('Ratio', fontsize=10)
    ax6.set_title('Frontier Expansion Ratio (Enqueued/Consumed)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    add_current_value_annotation(ax6, time, df['frontier_expansion_ratio'], '#7209B7')

    # ==================== Row 3: Cache & Lookup Performance ====================

    # 7. Cache Hit Rate
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(time, df['cache_hit_rate'], linewidth=2, color='#06A77D')
    ax7.set_xlabel('Time (seconds)', fontsize=10)
    ax7.set_ylabel('Cache Hit Rate (%)', fontsize=10)
    ax7.set_title('Lookup Cache Hit Rate', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 105])
    ax7.grid(True, alpha=0.3)
    add_current_value_annotation(ax7, time, df['cache_hit_rate'], '#06A77D', suffix='%')

    # 8. Lookup Hits vs Misses
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(time, df['lookup_hits'], label='Hits', linewidth=2, color='#06A77D')
    ax8.plot(time, df['lookup_misses'], label='Misses', linewidth=2, color='#D62246')
    ax8.plot(time, df['total_lookups'], label='Total', linewidth=2, color='#2E86AB', linestyle='--')
    ax8.set_xlabel('Time (seconds)', fontsize=10)
    ax8.set_ylabel('Count', fontsize=10)
    ax8.set_title('Lookup Operations', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    add_current_value_annotation(ax8, time, df['lookup_hits'], '#06A77D')
    add_current_value_annotation(ax8, time, df['lookup_misses'], '#D62246')

    # 9. Lookup Inserts
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(time, df['lookup_inserts'], linewidth=2, color='#7209B7')
    ax9.set_xlabel('Time (seconds)', fontsize=10)
    ax9.set_ylabel('Lookup Inserts', fontsize=10)
    ax9.set_title('Total Lookup Inserts', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    add_current_value_annotation(ax9, time, df['lookup_inserts'], '#7209B7')

    # ==================== Row 4: Game Outcomes & Ratios ====================

    # 10. Games Lost
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(time, df['games_lost'], linewidth=2, color='#D62246')
    ax10.set_xlabel('Time (seconds)', fontsize=10)
    ax10.set_ylabel('Games Lost', fontsize=10)
    ax10.set_title('Total Games Lost', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    add_current_value_annotation(ax10, time, df['games_lost'], '#D62246')

    # 11. Frontier Size vs Lookup Size Ratio
    ax11 = fig.add_subplot(gs[3, 1])
    frontier_to_lookup_ratio = df['frontier_size'] / df['lookup_size'].replace(0, 1)
    ax11.plot(time, frontier_to_lookup_ratio, linewidth=2, color='#A23B72')
    ax11.set_xlabel('Time (seconds)', fontsize=10)
    ax11.set_ylabel('Ratio', fontsize=10)
    ax11.set_title('Frontier/Lookup Size Ratio', fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3)
    add_current_value_annotation(ax11, time, frontier_to_lookup_ratio, '#A23B72')

    # 12. Efficiency Metrics
    ax12 = fig.add_subplot(gs[3, 2])
    # Calculate loss rate per board expanded
    loss_rate = (df['games_lost'] / df['boards_expanded'].replace(0, 1)) * 100
    ax12.plot(time, loss_rate, linewidth=2, color='#F18F01')
    ax12.set_xlabel('Time (seconds)', fontsize=10)
    ax12.set_ylabel('Loss Rate (%)', fontsize=10)
    ax12.set_title('Game Loss Rate (per Board Expanded)', fontsize=12, fontweight='bold')
    ax12.grid(True, alpha=0.3)
    add_current_value_annotation(ax12, time, loss_rate, '#F18F01', suffix='%')

    # Add overall title
    fig.suptitle('Tetris Atlas - RocksDB Statistics', fontsize=16, fontweight='bold', y=0.995)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Runtime: {df['timestamp_secs'].max():.1f} seconds ({df['timestamp_secs'].max()/60:.1f} minutes)")
    print(f"\nAtlas Size:")
    print(f"  Lookup Table: {df['lookup_size'].iloc[-1]:,} entries")
    print(f"  Frontier Queue: {df['frontier_size'].iloc[-1]:,} entries")
    print(f"  Frontier/Lookup Ratio: {(df['frontier_size'].iloc[-1] / max(df['lookup_size'].iloc[-1], 1)):.3f}")
    print(f"\nProgress:")
    print(f"  Boards Expanded: {df['boards_expanded'].iloc[-1]:,}")
    print(f"  Lookup Inserts: {df['lookup_inserts'].iloc[-1]:,}")
    print(f"  Games Lost: {df['games_lost'].iloc[-1]:,}")
    print(f"  Loss Rate: {(df['games_lost'].iloc[-1] / max(df['boards_expanded'].iloc[-1], 1) * 100):.2f}%")
    print(f"\nFrontier Operations:")
    print(f"  Enqueued: {df['frontier_enqueued'].iloc[-1]:,}")
    print(f"  Consumed: {df['frontier_consumed'].iloc[-1]:,}")
    print(f"  Deleted: {df['frontier_deleted'].iloc[-1]:,}")
    print(f"  Final Expansion Ratio: {df['frontier_expansion_ratio'].iloc[-1]:.3f}")
    print(f"\nLookup Performance:")
    print(f"  Lookup Hits: {df['lookup_hits'].iloc[-1]:,}")
    print(f"  Lookup Misses: {df['lookup_misses'].iloc[-1]:,}")
    print(f"  Total Lookups: {df['total_lookups'].iloc[-1]:,}")
    print(f"  Final Cache Hit Rate: {df['cache_hit_rate'].iloc[-1]:.2f}%")
    print(f"\nPerformance Rates:")
    print(f"  Avg Boards/sec: {df['boards_per_sec'].mean():.2f}")
    print(f"  Final Boards/sec: {df['boards_per_sec'].iloc[-1]:.2f}")
    print(f"  Avg Frontier Consumption: {df['frontier_consumption_rate'].mean():.2f}/sec")
    print(f"  Avg Frontier Expansion: {df['frontier_expansion_rate'].mean():.2f}/sec")
    print("="*60)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        print("\nDisplaying plot...")
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Tetris Atlas RocksDB statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot stats from default location
  python plot_tetris_atlas_stats_rocksdb.py

  # Specify CSV file and output path
  python plot_tetris_atlas_stats_rocksdb.py -i custom_stats.csv -o output.png

  # Use custom paths
  python plot_tetris_atlas_stats_rocksdb.py -i /path/to/stats.csv -o /path/to/plot.png
        """
    )

    parser.add_argument(
        '-i', '--input',
        default='tetris_atlas_stats_rocksdb.csv',
        help='Path to input CSV file (default: tetris_atlas_stats_rocksdb.csv)'
    )

    parser.add_argument(
        '-o', '--output',
        default='tetris_atlas_metrics_rocksdb.png',
        help='Path to output image file (default: tetris_atlas_metrics_rocksdb.png)'
    )

    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print(f"Current directory: {Path.cwd()}")
        return 1

    plot_tetris_atlas_stats_rocksdb(csv_path, args.output)
    return 0


if __name__ == '__main__':
    exit(main())
