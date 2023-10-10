use std::cmp::max;
use std::cmp::min;
use std::collections::HashMap;
use std::time::Instant;

use self::connect4::connect4_legal;
use self::connect4::connect4_move;
use self::connect4::connect4_new;
use self::connect4::Connect4;

mod connect4;

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

#[test]
fn test_mcts() {
    let game = connect4_new(7, 6);
}

#[test]
fn test_game_to_columns() {
    let mut game = connect4_new(7, 6);
    game.columns[0] = vec![false, true, false, true];
    game.columns[6] = vec![false, true, false, true, false];
    game.player = true;
    let result = game_to_node(&game, false);
    print!("{result:b}\n");
    assert_eq!(
        result,
        0b0111111000000000000000000000000000000001111001010000000000000000000000000000000001010
    );
    let new_game = node_to_game(result);
    let first = &new_game.columns[0];
    let last = &new_game.columns[6];
    print!("{first:?}\n");
    print!("{last:?}\n");
    assert_eq!(new_game.columns, game.columns);
}

fn game_to_node(game: &Connect4, terminal: bool) -> u128 {
    let mut bit_rep = 0 as u128;
    let mut bit_occupancy = 0 as u128;
    let mut counter = 0;
    for i in 0..game.columns.len() {
        for j in 0..game.height as usize {
            if game.columns[i].len() > j {
                bit_rep += (game.columns[i][j] as u128) << counter;
                bit_occupancy += 1 << counter;
            }
            counter += 1;
        }
    }
    bit_rep
        + (bit_occupancy << counter)
        + ((game.player as u128) << 2 * counter - 1)
        + ((terminal as u128) << 2 * counter)
}

fn node_to_game(node: u128) -> Connect4 {
    let terminal = &node >> 84;
    let player = ((&node >> 83) - (terminal << 1) & 1) == 1;
    let full_state = (&node - ((&node >> 83) << 83) - (terminal << 84)) >> 42;
    let player_state = &node - ((&node >> 83) << 83) - (terminal << 84) - (full_state << 42);
    print!("{player}\n{full_state:b}\n{player_state:b}\n");
    let mut board: Vec<Vec<bool>> = Vec::new();
    for i in 0..7 as usize {
        board.push(Vec::new());
        for j in 0..6 as usize {
            if (full_state >> (i * 6 + j) & 1) == 1 {
                board[i].push((player_state >> (i * 6 + j) & 1) == 1);
            }
        }
    }
    return Connect4 {
        columns: board,
        height: 6,
        player: player,
    };
}

// fn mcts(root: u8, tree: HashMap<u8, i32>) {
//     let mut time = 60;
//     while time > 0 {
//         let now = Instant::now();
//         let leaf = traverse(root);
//         let simulation_result = rollout(leaf);
//         backpropagate(leaf, simulation_result);
//         time -= now.elapsed().as_secs();
//     }
// }

fn traverse(
    root: u128,
    _tree: HashMap<u128, (Vec<u128>, f32)>,
    _total: u32,
    _record: HashMap<u128, u32>,
) -> (
    u128,
    u32,
    HashMap<u128, u32>,
    HashMap<u128, (Vec<u128>, f32)>,
) {
    let mut total = _total;
    let mut record = _record.clone();
    let mut leaf = 0;
    let mut fully_expanded = true;
    let mut node = root;
    let mut tree = _tree.clone();
    while fully_expanded {
        let mut children = tree.get(&node).unwrap().0.clone();
        if children.len() == 0 {
            for pos in 0..7 {
                let game = node_to_game(node);
                if connect4_legal(&game)[pos] {
                    let (terminal, new_game) = connect4_move(pos as u8, &game);
                    let new_node = game_to_node(&new_game, terminal);
                    tree.insert(new_node, (Vec::new(), 0.0));
                    children.push(new_node);
                    tree.insert(node, (children.clone(), tree.get(&node).unwrap().1));
                    record.insert(new_node, 0);
                }
            }
            leaf = children[0];
            fully_expanded = false;
        } else {
            let mut best_child = children[0];
            let mut best_value: f32 = 0.0;
            for child in &tree.get(&node).unwrap().0 {
                let ucb = tree.get(&child).unwrap().1
                    + ((total as f32).ln() / (record[&node] as f32)).sqrt();
                if ucb > best_value {
                    best_value = ucb;
                    best_child = *child;
                }
            }
            total += 1;
            record.insert(best_child, record.get(&best_child).unwrap() + 1);
            node = best_child;
        }
    }
    total += 1;
    record.insert(
        leaf,
        match record.get(&leaf) {
            Some(x) => x + 1,
            None => 1,
        },
    );
    (leaf, total, record, tree)
}
