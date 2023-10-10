#[derive(Clone)]
pub struct Connect4 {
    pub columns: Vec<Vec<bool>>,
    pub height: u8,
    pub player: bool,
}

pub fn connect4_new(num_columns: u8, height: u8) -> Connect4 {
    let mut columns = Vec::new();
    for _ in 0..num_columns {
        columns.push(Vec::<bool>::new());
    }
    Connect4 {
        columns: columns,
        height: height,
        player: false,
    }
}

pub fn connect4_legal(game: &Connect4) -> Vec<bool> {
    game.columns
        .iter()
        .map(|x| x.len() as u8 >= game.height)
        .collect()
}

fn check_contiguous(sequence: &Vec<bool>, player: bool) -> bool {
    let mut counter = 0;
    for i in 0..sequence.len() as usize {
        if sequence[i] == player {
            counter += 1;
            if counter >= 4 {
                return true;
            }
        } else {
            counter = 0;
        }
    }
    return false;
}

pub fn connect4_move(player_move: u8, game: &Connect4) -> (bool, Connect4) {
    let col = player_move as usize;
    let row = game.columns[col].len();
    if row as u8 > game.height {
        return (true, game.clone());
    }
    let mut end = false;
    let mut new = game.columns.clone();
    new[col].push(game.player);
    let end_state = Connect4 {
        columns: new,
        height: game.height,
        player: !game.player,
    };

    // check 3 under column
    if row > 2
        && end_state.columns[col][row - 1] == game.player
        && end_state.columns[col][row - 2] == game.player
        && end_state.columns[col][row - 3] == game.player
    {
        end = true;
    }

    // check row
    let query_row: &Vec<bool> = &end_state
        .columns
        .iter()
        .map(|x| if x.len() > row { x[row] } else { !game.player })
        .collect();
    if check_contiguous(&query_row, game.player) {
        end = true;
    }

    // check right ascending diagonal
    let rdiag_index = col as i8 - row as i8;
    let mut query_rdiag: Vec<bool> = Vec::new();
    for i in 0..game.columns.len() as usize {
        if i as i8 - rdiag_index >= 0 && end_state.columns[i].len() as i8 > (i as i8 - rdiag_index)
        {
            query_rdiag.push(end_state.columns[i][(i as i8 - rdiag_index) as usize]);
        } else {
            query_rdiag.push(!game.player);
        }
    }
    if check_contiguous(&query_rdiag, game.player) {
        end = true;
    }

    // check right ascending diagonal
    let ldiag_index = col + row;
    let mut query_ldiag: Vec<bool> = Vec::new();
    for i in 0..game.columns.len() as usize {
        if ldiag_index as i8 - i as i8 >= 0
            && end_state.columns[i].len() as i8 > ldiag_index as i8 - i as i8
        {
            query_ldiag.push(end_state.columns[i][ldiag_index - i]);
        } else {
            query_ldiag.push(!game.player);
        }
    }
    if check_contiguous(&query_ldiag, game.player) {
        end = true;
    }

    return (end, end_state);
}

#[test]
fn test_connect4() {
    let game = Connect4 {
        columns: vec![
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
        ],
        height: 6,
        player: false,
    };
    let step0 = connect4_move(0, &game);
    let step1 = connect4_move(0, &step0.1);
    let step2 = connect4_move(1, &step1.1);
    let step3 = connect4_move(1, &step2.1);
    let step4 = connect4_move(2, &step3.1);
    let step5 = connect4_move(2, &step4.1);
    let step6 = connect4_move(3, &step5.1);

    assert_eq!(step0.0, false);
    assert_eq!(step1.0, false);
    assert_eq!(step2.0, false);
    assert_eq!(step3.0, false);
    assert_eq!(step4.0, false);
    assert_eq!(step5.0, false);
    assert_eq!(step6.0, true);
}
