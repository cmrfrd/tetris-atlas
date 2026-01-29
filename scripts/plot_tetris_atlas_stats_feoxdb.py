#!/usr/bin/env python3
"""
Visualize Tetris Atlas FeoxDB Statistics

This script reads the tetris_atlas_stats.csv file and creates comprehensive
visualizations of all the key metrics including:
- Atlas expansion (boards, lookup table, frontier queue)
- Performance rates (boards/sec, frontier consumption/expansion)
- Cache performance
- Database operations and latencies
- Memory usage
- Write buffer statistics
- Disk I/O
- Error counts
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


def plot_tetris_atlas_stats(csv_path, output_path=None):
    """Create comprehensive visualization of Tetris Atlas statistics"""

    # Read CSV
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} data points")
    print(f"Time range: {df['timestamp_secs'].min():.1f}s to {df['timestamp_secs'].max():.1f}s")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)

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

    # 2. Boards Expanded
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, df['boards_expanded'], linewidth=2, color='#F18F01')
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('Boards Expanded', fontsize=10)
    ax2.set_title('Total Boards Expanded', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

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

    # ==================== Row 2: Performance Rates ====================

    # 4. Boards Per Second
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time, df['boards_per_sec'], linewidth=2, color='#F18F01')
    ax4.set_xlabel('Time (seconds)', fontsize=10)
    ax4.set_ylabel('Boards/sec', fontsize=10)
    ax4.set_title('Board Expansion Rate', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Frontier Consumption & Expansion Rates
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(time, df['frontier_consumption_rate'], label='Consumption', linewidth=2, color='#A23B72')
    ax5.plot(time, df['frontier_expansion_rate'], label='Expansion', linewidth=2, color='#06A77D')
    ax5.set_xlabel('Time (seconds)', fontsize=10)
    ax5.set_ylabel('Items/sec', fontsize=10)
    ax5.set_title('Frontier Queue Rates', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Frontier Expansion Ratio
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(time, df['frontier_expansion_ratio'], linewidth=2, color='#7209B7')
    ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ratio = 1.0')
    ax6.set_xlabel('Time (seconds)', fontsize=10)
    ax6.set_ylabel('Ratio', fontsize=10)
    ax6.set_title('Frontier Expansion Ratio (Enqueued/Consumed)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ==================== Row 3: Cache Performance ====================

    # 7. Cache Hit Rate
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(time, df['cache_hit_rate'], linewidth=2, color='#06A77D')
    ax7.set_xlabel('Time (seconds)', fontsize=10)
    ax7.set_ylabel('Cache Hit Rate (%)', fontsize=10)
    ax7.set_title('Lookup Cache Hit Rate', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 105])
    ax7.grid(True, alpha=0.3)

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

    # 9. DB Cache Hit Rate
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(time, df['db_cache_hit_rate'], linewidth=2, color='#7209B7')
    ax9_twin = ax9.twinx()
    ax9_twin.plot(time, df['db_cache_evictions'], label='Evictions', linewidth=2,
                  color='#F18F01', alpha=0.6, linestyle='--')
    ax9.set_xlabel('Time (seconds)', fontsize=10)
    ax9.set_ylabel('DB Cache Hit Rate (%)', fontsize=10, color='#7209B7')
    ax9_twin.set_ylabel('Cache Evictions', fontsize=10, color='#F18F01')
    ax9.set_title('Database Cache Performance', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9_twin.legend(loc='upper right')

    # ==================== Row 4: Database Operations ====================

    # 10. DB Operations Breakdown
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(time, df['db_total_gets'], label='Gets', linewidth=2, color='#2E86AB')
    ax10.plot(time, df['db_total_inserts'], label='Inserts', linewidth=2, color='#06A77D')
    ax10.plot(time, df['db_total_updates'], label='Updates', linewidth=2, color='#F18F01')
    ax10.plot(time, df['db_total_deletes'], label='Deletes', linewidth=2, color='#D62246')
    ax10.set_xlabel('Time (seconds)', fontsize=10)
    ax10.set_ylabel('Count', fontsize=10)
    ax10.set_title('Database Operations by Type', fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 11. Range Queries & Total Operations
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(time, df['db_total_range_queries'], label='Range Queries', linewidth=2, color='#A23B72')
    ax11_twin = ax11.twinx()
    ax11_twin.plot(time, df['db_total_operations'], label='Total Ops', linewidth=2,
                   color='#2E86AB', alpha=0.6, linestyle='--')
    ax11.set_xlabel('Time (seconds)', fontsize=10)
    ax11.set_ylabel('Range Queries', fontsize=10, color='#A23B72')
    ax11_twin.set_ylabel('Total Operations', fontsize=10, color='#2E86AB')
    ax11.set_title('Database Query Operations', fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3)
    ax11.legend(loc='upper left')
    ax11_twin.legend(loc='upper right')
    ax11.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax11_twin.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 12. Operation Latencies
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(time, df['db_avg_get_latency_ns'], label='Get', linewidth=2, color='#2E86AB')
    ax12.plot(time, df['db_avg_insert_latency_ns'], label='Insert', linewidth=2, color='#06A77D')
    ax12.plot(time, df['db_avg_delete_latency_ns'], label='Delete', linewidth=2, color='#D62246')
    ax12.set_xlabel('Time (seconds)', fontsize=10)
    ax12.set_ylabel('Latency (nanoseconds)', fontsize=10)
    ax12.set_title('Database Operation Latencies', fontsize=12, fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    # ==================== Row 5: Memory & Storage ====================

    # 13. Memory Usage
    ax13 = fig.add_subplot(gs[4, 0])
    ax13.plot(time, df['db_memory_usage'] / (1024**3), label='DB Memory', linewidth=2, color='#7209B7')
    ax13.plot(time, df['db_cache_memory'] / (1024**3), label='Cache Memory', linewidth=2, color='#A23B72')
    ax13.set_xlabel('Time (seconds)', fontsize=10)
    ax13.set_ylabel('Memory (GB)', fontsize=10)
    ax13.set_title('Memory Usage', fontsize=12, fontweight='bold')
    ax13.legend()
    ax13.grid(True, alpha=0.3)

    # 14. DB Record Count
    ax14 = fig.add_subplot(gs[4, 1])
    ax14.plot(time, df['db_record_count'], linewidth=2, color='#2E86AB')
    ax14.set_xlabel('Time (seconds)', fontsize=10)
    ax14.set_ylabel('Record Count', fontsize=10)
    ax14.set_title('Database Record Count', fontsize=12, fontweight='bold')
    ax14.grid(True, alpha=0.3)
    ax14.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 15. DB Cache Hits vs Misses
    ax15 = fig.add_subplot(gs[4, 2])
    ax15.plot(time, df['db_cache_hits'], label='Cache Hits', linewidth=2, color='#06A77D')
    ax15.plot(time, df['db_cache_misses'], label='Cache Misses', linewidth=2, color='#D62246')
    ax15.set_xlabel('Time (seconds)', fontsize=10)
    ax15.set_ylabel('Count', fontsize=10)
    ax15.set_title('Database Cache Hits & Misses', fontsize=12, fontweight='bold')
    ax15.legend()
    ax15.grid(True, alpha=0.3)
    ax15.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # ==================== Row 6: Write Buffer & Disk I/O ====================

    # 16. Write Buffer
    ax16 = fig.add_subplot(gs[5, 0])
    ax16.plot(time, df['db_writes_buffered'], label='Buffered', linewidth=2, color='#F18F01')
    ax16.plot(time, df['db_writes_flushed'], label='Flushed', linewidth=2, color='#06A77D')
    ax16_twin = ax16.twinx()
    ax16_twin.plot(time, df['db_flush_count'], label='Flush Count', linewidth=2,
                   color='#7209B7', alpha=0.6, linestyle='--')
    ax16.set_xlabel('Time (seconds)', fontsize=10)
    ax16.set_ylabel('Writes', fontsize=10)
    ax16_twin.set_ylabel('Flush Count', fontsize=10, color='#7209B7')
    ax16.set_title('Write Buffer Statistics', fontsize=12, fontweight='bold')
    ax16.legend(loc='upper left')
    ax16_twin.legend(loc='upper right')
    ax16.grid(True, alpha=0.3)
    ax16.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 17. Disk I/O Operations
    ax17 = fig.add_subplot(gs[5, 1])
    ax17.plot(time, df['db_disk_reads'], label='Disk Reads', linewidth=2, color='#2E86AB')
    ax17.plot(time, df['db_disk_writes'], label='Disk Writes', linewidth=2, color='#D62246')
    ax17.set_xlabel('Time (seconds)', fontsize=10)
    ax17.set_ylabel('Operations', fontsize=10)
    ax17.set_title('Disk I/O Operations', fontsize=12, fontweight='bold')
    ax17.legend()
    ax17.grid(True, alpha=0.3)

    # 18. Disk I/O Bytes
    ax18 = fig.add_subplot(gs[5, 2])
    ax18.plot(time, df['db_disk_bytes_read'] / (1024**3), label='Read (GB)', linewidth=2, color='#2E86AB')
    ax18.plot(time, df['db_disk_bytes_written'] / (1024**3), label='Written (GB)', linewidth=2, color='#D62246')
    ax18.set_xlabel('Time (seconds)', fontsize=10)
    ax18.set_ylabel('Data (GB)', fontsize=10)
    ax18.set_title('Disk I/O Throughput', fontsize=12, fontweight='bold')
    ax18.legend()
    ax18.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle('Tetris Atlas - FeoxDB Statistics', fontsize=16, fontweight='bold', y=0.995)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Runtime: {df['timestamp_secs'].max():.1f} seconds ({df['timestamp_secs'].max()/60:.1f} minutes)")
    print(f"\nAtlas Size:")
    print(f"  Lookup Table: {df['lookup_size'].iloc[-1]:,} entries")
    print(f"  Frontier Queue: {df['frontier_size'].iloc[-1]:,} entries")
    print(f"  DB Records: {df['db_record_count'].iloc[-1]:,}")
    print(f"\nProgress:")
    print(f"  Boards Expanded: {df['boards_expanded'].iloc[-1]:,}")
    print(f"  Lookup Inserts: {df['lookup_inserts'].iloc[-1]:,}")
    print(f"  Games Lost: {df['games_lost'].iloc[-1]:,}")
    print(f"\nFrontier Operations:")
    print(f"  Enqueued: {df['frontier_enqueued'].iloc[-1]:,}")
    print(f"  Consumed: {df['frontier_consumed'].iloc[-1]:,}")
    print(f"  Deleted: {df['frontier_deleted'].iloc[-1]:,}")
    print(f"\nPerformance:")
    print(f"  Avg Boards/sec: {df['boards_per_sec'].mean():.2f}")
    print(f"  Final Cache Hit Rate: {df['cache_hit_rate'].iloc[-1]:.2f}%")
    print(f"  Final Frontier Expansion Ratio: {df['frontier_expansion_ratio'].iloc[-1]:.3f}")
    print(f"\nMemory:")
    print(f"  DB Memory: {human_readable_size(df['db_memory_usage'].iloc[-1])}")
    print(f"  Cache Memory: {human_readable_size(df['db_cache_memory'].iloc[-1])}")
    print(f"\nDatabase Operations:")
    print(f"  Total Gets: {df['db_total_gets'].iloc[-1]:,}")
    print(f"  Total Inserts: {df['db_total_inserts'].iloc[-1]:,}")
    print(f"  Total Range Queries: {df['db_total_range_queries'].iloc[-1]:,}")
    print(f"  DB Cache Hit Rate: {df['db_cache_hit_rate'].iloc[-1]:.2f}%")
    print(f"\nWrite Buffer:")
    print(f"  Buffered: {df['db_writes_buffered'].iloc[-1]:,}")
    print(f"  Flushed: {df['db_writes_flushed'].iloc[-1]:,}")
    print(f"  Flush Count: {df['db_flush_count'].iloc[-1]:,}")
    print(f"  Write Failures: {df['db_write_failures'].iloc[-1]:,}")

    # Check for disk I/O
    has_disk_io = (df['db_disk_reads'].iloc[-1] > 0 or df['db_disk_writes'].iloc[-1] > 0)
    if has_disk_io:
        print(f"\nDisk I/O:")
        print(f"  Reads: {df['db_disk_reads'].iloc[-1]:,} ({human_readable_size(df['db_disk_bytes_read'].iloc[-1])})")
        print(f"  Writes: {df['db_disk_writes'].iloc[-1]:,} ({human_readable_size(df['db_disk_bytes_written'].iloc[-1])})")
    else:
        print(f"\nDisk I/O: None (all operations in memory)")

    # Check for errors
    total_errors = (df['db_key_not_found_errors'].iloc[-1] +
                   df['db_out_of_memory_errors'].iloc[-1] +
                   df['db_io_errors'].iloc[-1])
    if total_errors > 0:
        print(f"\nErrors:")
        print(f"  Key Not Found: {df['db_key_not_found_errors'].iloc[-1]:,}")
        print(f"  Out of Memory: {df['db_out_of_memory_errors'].iloc[-1]:,}")
        print(f"  I/O Errors: {df['db_io_errors'].iloc[-1]:,}")
    else:
        print(f"\nErrors: None")

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
        description='Visualize Tetris Atlas FeoxDB statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot stats from default location
  python plot_tetris_atlas_stats_feoxdb.py

  # Specify CSV file and output path
  python plot_tetris_atlas_stats_feoxdb.py -i custom_stats.csv -o output.png

  # Use custom paths
  python plot_tetris_atlas_stats_feoxdb.py -i /path/to/stats.csv -o /path/to/plot.png
        """
    )

    parser.add_argument(
        '-i', '--input',
        default='tetris_atlas_stats.csv',
        help='Path to input CSV file (default: tetris_atlas_stats.csv)'
    )

    parser.add_argument(
        '-o', '--output',
        default='tetris_atlas_metrics_feoxdb.png',
        help='Path to output image file (default: tetris_atlas_metrics_feoxdb.png)'
    )

    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print(f"Current directory: {Path.cwd()}")
        return 1

    plot_tetris_atlas_stats(csv_path, args.output)
    return 0


if __name__ == '__main__':
    exit(main())
