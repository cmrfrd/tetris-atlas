//! TUI (Text User Interface) for playing Tetris
//!
//! This module provides an interactive terminal-based interface for playing Tetris
//! using ratatui and crossterm.

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols::border,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

use crate::tetris::{TetrisGame, TetrisPiece};

/// Colors for each piece type (matching classic Tetris)
fn piece_color(piece: TetrisPiece) -> Color {
    match piece {
        p if p == TetrisPiece::O_PIECE => Color::Yellow,
        p if p == TetrisPiece::I_PIECE => Color::Cyan,
        p if p == TetrisPiece::S_PIECE => Color::Green,
        p if p == TetrisPiece::Z_PIECE => Color::Red,
        p if p == TetrisPiece::T_PIECE => Color::Magenta,
        p if p == TetrisPiece::L_PIECE => Color::Rgb(255, 165, 0), // Orange
        p if p == TetrisPiece::J_PIECE => Color::Blue,
        _ => Color::White,
    }
}

/// Get the shape of a piece as a 4x4 grid for display
/// These match the game's piece definitions in tetris_piece_data
fn piece_shape(piece: TetrisPiece, rotation: u8) -> [[bool; 4]; 4] {
    let mut grid = [[false; 4]; 4];

    match piece {
        p if p == TetrisPiece::O_PIECE => {
            // O piece - same in all rotations
            // 11
            // 11
            grid[0][0] = true;
            grid[0][1] = true;
            grid[1][0] = true;
            grid[1][1] = true;
        }
        p if p == TetrisPiece::I_PIECE => {
            match rotation % 2 {
                0 => {
                    // Horizontal: 1111
                    grid[0][0] = true;
                    grid[0][1] = true;
                    grid[0][2] = true;
                    grid[0][3] = true;
                }
                _ => {
                    // Vertical: 1
                    //           1
                    //           1
                    //           1
                    grid[0][0] = true;
                    grid[1][0] = true;
                    grid[2][0] = true;
                    grid[3][0] = true;
                }
            }
        }
        p if p == TetrisPiece::S_PIECE => match rotation % 2 {
            0 => {
                // 011
                // 110
                grid[0][1] = true;
                grid[0][2] = true;
                grid[1][0] = true;
                grid[1][1] = true;
            }
            _ => {
                // 10
                // 11
                // 01
                grid[0][0] = true;
                grid[1][0] = true;
                grid[1][1] = true;
                grid[2][1] = true;
            }
        },
        p if p == TetrisPiece::Z_PIECE => match rotation % 2 {
            0 => {
                // 110
                // 011
                grid[0][0] = true;
                grid[0][1] = true;
                grid[1][1] = true;
                grid[1][2] = true;
            }
            _ => {
                // 01
                // 11
                // 10
                grid[0][1] = true;
                grid[1][0] = true;
                grid[1][1] = true;
                grid[2][0] = true;
            }
        },
        p if p == TetrisPiece::T_PIECE => match rotation % 4 {
            0 => {
                // 111
                // 010
                grid[0][0] = true;
                grid[0][1] = true;
                grid[0][2] = true;
                grid[1][1] = true;
            }
            1 => {
                // 01
                // 11
                // 01
                grid[0][1] = true;
                grid[1][0] = true;
                grid[1][1] = true;
                grid[2][1] = true;
            }
            2 => {
                // 010
                // 111
                grid[0][1] = true;
                grid[1][0] = true;
                grid[1][1] = true;
                grid[1][2] = true;
            }
            _ => {
                // 10
                // 11
                // 10
                grid[0][0] = true;
                grid[1][0] = true;
                grid[1][1] = true;
                grid[2][0] = true;
            }
        },
        p if p == TetrisPiece::L_PIECE => match rotation % 4 {
            0 => {
                // 001
                // 111
                grid[0][2] = true;
                grid[1][0] = true;
                grid[1][1] = true;
                grid[1][2] = true;
            }
            1 => {
                // 10
                // 10
                // 11
                grid[0][0] = true;
                grid[1][0] = true;
                grid[2][0] = true;
                grid[2][1] = true;
            }
            2 => {
                // 111
                // 100
                grid[0][0] = true;
                grid[0][1] = true;
                grid[0][2] = true;
                grid[1][0] = true;
            }
            _ => {
                // 11
                // 01
                // 01
                grid[0][0] = true;
                grid[0][1] = true;
                grid[1][1] = true;
                grid[2][1] = true;
            }
        },
        p if p == TetrisPiece::J_PIECE => match rotation % 4 {
            0 => {
                // 100
                // 111
                grid[0][0] = true;
                grid[1][0] = true;
                grid[1][1] = true;
                grid[1][2] = true;
            }
            1 => {
                // 11
                // 10
                // 10
                grid[0][0] = true;
                grid[0][1] = true;
                grid[1][0] = true;
                grid[2][0] = true;
            }
            2 => {
                // 111
                // 001
                grid[0][0] = true;
                grid[0][1] = true;
                grid[0][2] = true;
                grid[1][2] = true;
            }
            _ => {
                // 01
                // 01
                // 11
                grid[0][1] = true;
                grid[1][1] = true;
                grid[2][0] = true;
                grid[2][1] = true;
            }
        },
        _ => {}
    }
    grid
}

/// Application state for the TUI
pub struct TetrisTui {
    game: TetrisGame,
    /// Index into the sorted list of valid placements
    selected_index: usize,
    game_over: bool,
    message: Option<(String, Instant)>,
    high_score: u32,
}

impl TetrisTui {
    pub fn new() -> Self {
        Self {
            game: TetrisGame::new(),
            selected_index: 0,
            game_over: false,
            message: None,
            high_score: 0,
        }
    }

    fn set_message(&mut self, msg: &str) {
        self.message = Some((msg.to_string(), Instant::now()));
    }

    fn get_message(&self) -> Option<&str> {
        self.message.as_ref().and_then(|(msg, time)| {
            if time.elapsed() < Duration::from_secs(2) {
                Some(msg.as_str())
            } else {
                None
            }
        })
    }

    /// Get the currently selected placement (always valid)
    fn current_placement(&self) -> Option<&crate::tetris::TetrisPiecePlacement> {
        let placements = self.game.current_placements();
        if placements.is_empty() {
            None
        } else {
            Some(&placements[self.selected_index % placements.len()])
        }
    }

    /// Get column of current selection
    fn selected_column(&self) -> u8 {
        self.current_placement()
            .map(|p| {
                // Extract column value via Display format
                let s = format!("{}", p.orientation.column);
                s.trim_start_matches("Column(")
                    .trim_end_matches(")")
                    .parse()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    /// Get rotation of current selection
    fn selected_rotation(&self) -> u8 {
        self.current_placement()
            .map(|p| {
                let s = format!("{}", p.orientation.rotation);
                s.parse().unwrap_or(0)
            })
            .unwrap_or(0)
    }

    fn apply_current_placement(&mut self) {
        if self.game_over {
            return;
        }

        if let Some(&placement) = self.current_placement() {
            let is_lost = self.game.apply_placement(placement);
            if is_lost.into() {
                self.game_over = true;
                if self.game.lines_cleared > self.high_score {
                    self.high_score = self.game.lines_cleared;
                }
                self.set_message("GAME OVER! Press 'r' to restart");
            } else {
                // Reset selection for new piece, try to keep similar column
                self.select_nearest_column(self.selected_column());
            }
        }
    }

    /// Select the placement nearest to the given column
    fn select_nearest_column(&mut self, target_col: u8) {
        let placements = self.game.current_placements();
        if placements.is_empty() {
            self.selected_index = 0;
            return;
        }

        // Find placement with column closest to target
        let mut best_index = 0;
        let mut best_diff = u8::MAX;
        for (i, p) in placements.iter().enumerate() {
            let col_str = format!("{}", p.orientation.column);
            let col: u8 = col_str
                .trim_start_matches("Column(")
                .trim_end_matches(")")
                .parse()
                .unwrap_or(0);
            let diff = col.abs_diff(target_col);
            if diff < best_diff {
                best_diff = diff;
                best_index = i;
            }
        }
        self.selected_index = best_index;
    }

    fn restart(&mut self) {
        if self.game.lines_cleared > self.high_score {
            self.high_score = self.game.lines_cleared;
        }
        self.game = TetrisGame::new();
        self.selected_index = 0;
        self.select_nearest_column(4); // Start near middle
        self.game_over = false;
        self.message = None;
    }

    fn move_left(&mut self) {
        let placements = self.game.current_placements();
        if !placements.is_empty() {
            // Move to previous placement (wrapping)
            if self.selected_index == 0 {
                self.selected_index = placements.len() - 1;
            } else {
                self.selected_index -= 1;
            }
        }
    }

    fn move_right(&mut self) {
        let placements = self.game.current_placements();
        if !placements.is_empty() {
            // Move to next placement (wrapping)
            self.selected_index = (self.selected_index + 1) % placements.len();
        }
    }

    fn rotate(&mut self) {
        // Find next placement with same column but different rotation
        let current_col = self.selected_column();
        let current_rot = self.selected_rotation();
        let placements = self.game.current_placements();

        // Look for next rotation at same column
        for i in 1..=placements.len() {
            let idx = (self.selected_index + i) % placements.len();
            let p = &placements[idx];
            let col_str = format!("{}", p.orientation.column);
            let col: u8 = col_str
                .trim_start_matches("Column(")
                .trim_end_matches(")")
                .parse()
                .unwrap_or(0);
            let rot_str = format!("{}", p.orientation.rotation);
            let rot: u8 = rot_str.parse().unwrap_or(0);

            if col == current_col && rot != current_rot {
                self.selected_index = idx;
                return;
            }
        }
        // If no other rotation at same column, just go to next
        self.move_right();
    }

    fn rotate_back(&mut self) {
        // Find previous placement with same column but different rotation
        let current_col = self.selected_column();
        let current_rot = self.selected_rotation();
        let placements = self.game.current_placements();

        // Look for previous rotation at same column
        for i in 1..=placements.len() {
            let idx = (self.selected_index + placements.len() - i) % placements.len();
            let p = &placements[idx];
            let col_str = format!("{}", p.orientation.column);
            let col: u8 = col_str
                .trim_start_matches("Column(")
                .trim_end_matches(")")
                .parse()
                .unwrap_or(0);
            let rot_str = format!("{}", p.orientation.rotation);
            let rot: u8 = rot_str.parse().unwrap_or(0);

            if col == current_col && rot != current_rot {
                self.selected_index = idx;
                return;
            }
        }
        // If no other rotation at same column, just go to previous
        self.move_left();
    }
}

/// Run the TUI application
pub fn run() -> anyhow::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = TetrisTui::new();

    // Main loop
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut TetrisTui,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Left | KeyCode::Char('a') | KeyCode::Char('h') => app.move_left(),
                        KeyCode::Right | KeyCode::Char('d') | KeyCode::Char('l') => {
                            app.move_right()
                        }
                        KeyCode::Up | KeyCode::Char('w') | KeyCode::Char('k') => app.rotate(),
                        KeyCode::Down | KeyCode::Char('s') | KeyCode::Char('j') => {
                            app.rotate_back()
                        }
                        KeyCode::Enter | KeyCode::Char(' ') => app.apply_current_placement(),
                        KeyCode::Char('r') => app.restart(),
                        _ => {}
                    }
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &TetrisTui) {
    let size = f.area();

    // Main layout: sidebar | board | info
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(20), // Left panel (next piece + controls)
            Constraint::Length(24), // Board (10 cols * 2 chars + borders)
            Constraint::Min(20),    // Right panel (stats)
        ])
        .split(size);

    // Left panel layout
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Current piece preview
            Constraint::Length(8), // Next piece preview
            Constraint::Min(10),   // Controls
        ])
        .split(main_chunks[0]);

    // Draw current piece
    draw_piece_preview(
        f,
        left_chunks[0],
        app.game.current_piece,
        app.selected_rotation(),
        "Current",
    );

    // Draw next piece
    let next_piece = app.game.peek_nth_next_piece(1);
    draw_piece_preview(f, left_chunks[1], next_piece, 0, "Next");

    // Draw controls
    draw_controls(f, left_chunks[2]);

    // Draw game board
    draw_board(f, main_chunks[1], app);

    // Right panel layout
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Stats
            Constraint::Length(6),  // Placement info
            Constraint::Min(5),     // Message
        ])
        .split(main_chunks[2]);

    // Draw stats
    draw_stats(f, right_chunks[0], app);

    // Draw placement info
    draw_placement_info(f, right_chunks[1], app);

    // Draw message
    if let Some(msg) = app.get_message() {
        let msg_block = Block::default()
            .borders(Borders::ALL)
            .border_set(border::ROUNDED)
            .title(" Message ")
            .style(Style::default().fg(Color::Yellow));
        let msg_para = Paragraph::new(msg)
            .block(msg_block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Yellow).bold());
        f.render_widget(msg_para, right_chunks[2]);
    }

    // Game over overlay
    if app.game_over {
        draw_game_over(f, size, app);
    }
}

fn draw_piece_preview(f: &mut Frame, area: Rect, piece: TetrisPiece, rotation: u8, title: &str) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(border::ROUNDED)
        .title(format!(" {} ", title))
        .style(Style::default().fg(piece_color(piece)));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let shape = piece_shape(piece, rotation);
    let color = piece_color(piece);

    let mut lines: Vec<Line> = Vec::new();
    for row in 0..4 {
        let mut spans: Vec<Span> = Vec::new();
        for col in 0..4 {
            if shape[row][col] {
                spans.push(Span::styled("██", Style::default().fg(color)));
            } else {
                spans.push(Span::raw("  "));
            }
        }
        lines.push(Line::from(spans));
    }

    let preview = Paragraph::new(lines).alignment(Alignment::Center);
    f.render_widget(preview, inner);
}

fn draw_controls(f: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(border::ROUNDED)
        .title(" Controls ")
        .style(Style::default().fg(Color::Rgb(150, 150, 150)));

    let controls = vec![
        Line::from(vec![
            Span::styled("←/→ ", Style::default().fg(Color::Cyan).bold()),
            Span::raw("Move"),
        ]),
        Line::from(vec![
            Span::styled("↑/↓ ", Style::default().fg(Color::Cyan).bold()),
            Span::raw("Rotate"),
        ]),
        Line::from(vec![
            Span::styled("Enter", Style::default().fg(Color::Green).bold()),
            Span::raw(" Place"),
        ]),
        Line::from(vec![
            Span::styled("r   ", Style::default().fg(Color::Yellow).bold()),
            Span::raw(" Restart"),
        ]),
        Line::from(vec![
            Span::styled("q   ", Style::default().fg(Color::Red).bold()),
            Span::raw(" Quit"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "WASD/HJKL",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "also work!",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let para = Paragraph::new(controls).block(block);
    f.render_widget(para, area);
}

/// Calculate where the ghost piece would be displayed
/// Returns a set of (col, row) positions for the ghost piece
fn calculate_ghost_piece(app: &TetrisTui) -> Vec<(usize, usize)> {
    if app.game_over {
        return vec![];
    }

    if app.current_placement().is_none() {
        return vec![];
    }

    let shape = piece_shape(app.game.current_piece, app.selected_rotation());
    let board = &app.game.board;
    let selected_col = app.selected_column() as i32;

    // Find the piece's cells in the 4x4 grid
    // Note: In the grid, row 0 is TOP, row 3 is BOTTOM (visually)
    // On the board, row 0 is BOTTOM, row 19 is TOP
    let mut piece_cells: Vec<(i32, i32)> = vec![];
    for grid_row in 0..4 {
        for grid_col in 0..4 {
            if shape[grid_row][grid_col] {
                piece_cells.push((grid_col as i32, grid_row as i32));
            }
        }
    }

    // All tetrominoes have exactly 4 cells
    if piece_cells.len() != 4 {
        return vec![];
    }

    // Find the bounding box of the piece in the grid
    let min_piece_col = piece_cells.iter().map(|(c, _)| *c).min().unwrap_or(0);
    let max_grid_row = piece_cells.iter().map(|(_, r)| *r).max().unwrap_or(0);

    // Calculate board column for each piece cell
    // The selected_column is where the LEFT edge of the piece goes
    let get_board_col = |piece_col: i32| -> i32 { selected_col + (piece_col - min_piece_col) };

    // Verify all columns are valid
    for &(pc, _) in &piece_cells {
        let board_col = get_board_col(pc);
        if board_col < 0 || board_col >= 10 {
            return vec![];
        }
    }

    // Find the landing position by checking each row from bottom to top
    // The piece lands at the lowest row where it fits and would collide if dropped lower
    for drop_row in 0i32..20 {
        // Calculate where each piece cell would be at this drop position
        let mut positions = Vec::with_capacity(4);
        let mut all_valid = true;

        for &(pc, pr) in &piece_cells {
            let board_col = get_board_col(pc);
            // Grid row 0 = top of piece shape, max_grid_row = bottom of piece shape
            // The bottom of the piece (max_grid_row) sits at drop_row
            // Higher grid rows (lower in piece) go to lower board rows
            let board_row = drop_row + (max_grid_row - pr);

            if board_col < 0 || board_col >= 10 || board_row < 0 || board_row >= 20 {
                all_valid = false;
                break;
            }

            // Check collision with existing blocks
            if board.get_bit(board_col as usize, board_row as usize) {
                all_valid = false;
                break;
            }

            positions.push((board_col as usize, board_row as usize));
        }

        if !all_valid || positions.len() != 4 {
            continue;
        }

        // Check if piece would collide one row lower (meaning this is the landing spot)
        let is_landing_spot = if drop_row == 0 {
            true // Already at bottom of board
        } else {
            // Check if any piece cell would collide at drop_row - 1
            piece_cells.iter().any(|&(pc, pr)| {
                let board_col = get_board_col(pc);
                let board_row = (drop_row - 1) + (max_grid_row - pr);
                if board_row < 0 {
                    true // Would be below the board
                } else {
                    board.get_bit(board_col as usize, board_row as usize)
                }
            })
        };

        if is_landing_spot {
            return positions;
        }
    }

    vec![]
}

fn draw_board(f: &mut Frame, area: Rect, app: &TetrisTui) {
    let board = &app.game.board;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(border::DOUBLE)
        .title(" TETRIS ")
        .title_style(Style::default().fg(Color::Cyan).bold())
        .style(Style::default().fg(Color::White));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Calculate ghost piece positions
    let ghost_positions = calculate_ghost_piece(app);
    let ghost_color = piece_color(app.game.current_piece);

    // We show rows 0-19 (bottom to top)
    let display_rows = 20;
    let mut lines: Vec<Line> = Vec::new();

    for row in (0..display_rows).rev() {
        let mut spans: Vec<Span> = Vec::new();
        for col in 0..10 {
            let bit = board.get_bit(col, row);
            if bit {
                // Filled cell
                spans.push(Span::styled(
                    "██",
                    Style::default().fg(Color::Rgb(100, 100, 100)),
                ));
            } else if ghost_positions.contains(&(col, row)) {
                // Ghost piece cell
                spans.push(Span::styled("▒▒", Style::default().fg(ghost_color)));
            } else {
                spans.push(Span::styled(
                    "··",
                    Style::default().fg(Color::Rgb(40, 40, 40)),
                ));
            }
        }
        lines.push(Line::from(spans));
    }

    // Add column numbers at bottom
    let mut col_nums: Vec<Span> = Vec::new();
    for col in 0..10 {
        let style = if col as u8 == app.selected_column() {
            Style::default().fg(Color::Cyan).bold()
        } else {
            Style::default().fg(Color::DarkGray)
        };
        col_nums.push(Span::styled(format!("{} ", col), style));
    }
    lines.push(Line::from(col_nums));

    let board_para = Paragraph::new(lines);
    f.render_widget(board_para, inner);
}

fn draw_stats(f: &mut Frame, area: Rect, app: &TetrisTui) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(border::ROUNDED)
        .title(" Stats ")
        .style(Style::default().fg(Color::Magenta));

    let stats = vec![
        Line::from(vec![
            Span::styled("Lines: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{}", app.game.lines_cleared),
                Style::default().fg(Color::Green).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("Pieces: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{}", app.game.piece_count),
                Style::default().fg(Color::Cyan).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("High Score: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{}", app.high_score),
                Style::default().fg(Color::Yellow).bold(),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Piece: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{}", app.game.current_piece),
                Style::default()
                    .fg(piece_color(app.game.current_piece))
                    .bold(),
            ),
        ]),
    ];

    let para = Paragraph::new(stats).block(block);
    f.render_widget(para, area);
}

fn draw_placement_info(f: &mut Frame, area: Rect, app: &TetrisTui) {
    let num_placements = app.game.current_placements().len();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(border::ROUNDED)
        .title(" Placement ")
        .style(Style::default().fg(Color::Green));

    let info = vec![
        Line::from(vec![
            Span::styled("Column: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{}", app.selected_column()),
                Style::default().fg(Color::Cyan).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("Rotation: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{}", app.selected_rotation()),
                Style::default().fg(Color::Cyan).bold(),
            ),
        ]),
        Line::from(vec![Span::styled(
            format!("{}/{}", app.selected_index + 1, num_placements),
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    let para = Paragraph::new(info).block(block);
    f.render_widget(para, area);
}

fn draw_game_over(f: &mut Frame, area: Rect, app: &TetrisTui) {
    let popup_width = 40;
    let popup_height = 10;
    let popup_area = Rect {
        x: area.width.saturating_sub(popup_width) / 2,
        y: area.height.saturating_sub(popup_height) / 2,
        width: popup_width.min(area.width),
        height: popup_height.min(area.height),
    };

    // Clear the area behind the popup
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(border::DOUBLE)
        .title(" GAME OVER ")
        .title_style(Style::default().fg(Color::Red).bold())
        .style(Style::default().bg(Color::Rgb(20, 20, 30)).fg(Color::Red));

    let content = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  ╔═══════════════════════╗  ",
            Style::default().fg(Color::Red),
        )),
        Line::from(Span::styled(
            format!("  ║   Lines: {:>6}       ║  ", app.game.lines_cleared),
            Style::default().fg(Color::Yellow).bold(),
        )),
        Line::from(Span::styled(
            format!("  ║   Pieces: {:>5}       ║  ", app.game.piece_count),
            Style::default().fg(Color::Cyan),
        )),
        Line::from(Span::styled(
            "  ╚═══════════════════════╝  ",
            Style::default().fg(Color::Red),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "   Press 'r' to restart",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::SLOW_BLINK),
        )),
    ];

    let para = Paragraph::new(content)
        .block(block)
        .alignment(Alignment::Center);
    f.render_widget(para, popup_area);
}
