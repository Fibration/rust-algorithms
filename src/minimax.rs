use std::cmp::max;
use std::cmp::min;

#[cfg(test)]
mod test_minimax {
    use super::*;

    #[test]
    fn test_terminal_state() {
        let node = (0, 1, String::from(""));
        let tree = vec![node.clone()];
        assert_eq!(minimax(node.clone(), tree, true), 1)
    }

    #[test]
    fn test_two_steps() {
        let node1 = (0, 1, String::from("1"));
        let node2 = (1, -1, String::from(""));
        let tree = vec![node1.clone(), node2.clone()];
        assert_eq!(minimax(node1.clone(), tree, true), -1)
    }

    #[test]
    fn test_choice() {
        let node1 = (0, 0, String::from("1,2"));
        let node2 = (1, 1, String::from("3,4"));
        let node3 = (2, 1, String::from("5,6"));
        let node2a = (3, 1, String::from(""));
        let node2b = (4, 2, String::from(""));
        let node3a = (5, 3, String::from(""));
        let node3b = (6, 4, String::from(""));
        let tree = vec![
            node1.clone(),
            node2.clone(),
            node3.clone(),
            node2a.clone(),
            node2b.clone(),
            node3a.clone(),
            node3b.clone(),
        ];
        assert_eq!(minimax(node1.clone(), tree.clone(), true), 3);
        assert_eq!(negamax(node1.clone(), tree.clone()), 3);
        assert_eq!(negamax_alphabeta(&node1, &tree, 0, 100, 5), 3);
        print!("MTD\n");
        assert_eq!(mtd(&node1, &tree, 0, 5), 3);
    }
}

#[cfg(test)]
mod connect4 {

    struct Connect4 {
        columns: Vec<Vec<bool>>,
        player: bool,
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

    fn connect4_move(player_move: u8, game: &Connect4) -> (bool, Connect4) {
        let col = player_move as usize;
        let row = game.columns[col].len();
        let mut end = false;
        let mut new = game.columns.clone();
        new[col].push(game.player);
        let end_state = Connect4 {
            columns: new,
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
            if i as i8 - rdiag_index >= 0
                && end_state.columns[i].len() as i8 > (i as i8 - rdiag_index)
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
}

// 30/09/2023
pub fn minimax(node: (u32, i32, String), tree: Vec<(u32, i32, String)>, maximize: bool) -> i32 {
    let node_id = node.0;
    print!("{node_id}, {maximize};\n");
    if node.2.len() == 0 {
        return node.1;
    } else {
        let children: Vec<u32> = node
            .2
            .split(",")
            .into_iter()
            .map(|x| x.parse().unwrap())
            .collect();
        let new_tree = tree.clone();
        let nodes = new_tree
            .iter()
            .filter(|x| children.contains(&x.0))
            .collect::<Vec<&(u32, i32, String)>>();
        if maximize {
            let mut value = -9999;
            for child_node in nodes {
                value = max(value, minimax(child_node.clone(), tree.clone(), false))
            }
            return value;
        } else {
            let mut value = 9999;

            for child_node in nodes {
                value = min(value, minimax(child_node.clone(), tree.clone(), true))
            }
            return value;
        }
    }
}

fn get_nodes_by_id(node_ids: Vec<u32>, tree: &Vec<(u32, i32, String)>) -> Vec<&(u32, i32, String)> {
    let nodes = tree
        .iter()
        .filter(|x| node_ids.contains(&x.0))
        .collect::<Vec<&(u32, i32, String)>>();
    nodes
}

fn string_to_idx(string: String) -> Vec<u32> {
    string
        .split(",")
        .into_iter()
        .map(|x| x.parse().unwrap())
        .collect()
}

// 01/09/2023
fn negamax(node: (u32, i32, String), tree: Vec<(u32, i32, String)>) -> i32 {
    if node.2.len() == 0 {
        return node.1;
    } else {
        let node_ids = string_to_idx(node.2);
        let new_tree = tree.clone();
        let nodes = get_nodes_by_id(node_ids, &new_tree);

        let mut value = -9999;
        for child_node in nodes {
            value = max(value, -negamax(child_node.clone(), tree.clone()))
        }
        return value;
    }
}

// 02/09/2023
fn negamax_alphabeta(
    node: &(u32, i32, String),
    tree: &Vec<(u32, i32, String)>,
    alpha: i32,
    beta: i32,
    depth: u8,
) -> i32 {
    let mut a = alpha;
    if depth == 0 || node.2.len() == 0 {
        return node.1;
    } else {
        let mut value = -9999;
        for child_node in get_nodes_by_id(string_to_idx(node.2.clone()), tree) {
            value = max(
                value,
                -negamax_alphabeta(child_node, tree, -beta, -a, depth - 1),
            );
            print!("value:{value},alpha:{a},beta:{beta},depth:{depth};\n");
            a = max(a, value);
            if a >= beta {
                break;
            }
        }
        return value;
    }
}

// 03/09/2023
// typically, you'd memoize the result but this mock tree is basically a memoize result anyway oops
fn mtd(node: &(u32, i32, String), tree: &Vec<(u32, i32, String)>, guess: i32, depth: u8) -> i32 {
    let mut g = guess;
    let mut upper = 9999;
    let mut lower = -9999;

    while lower < upper {
        let beta = max(g, lower + 1);
        g = negamax_alphabeta(node, tree, beta - 1, beta, depth);
        if g < beta {
            upper = g;
        } else {
            lower = g;
        }
    }
    g
}
