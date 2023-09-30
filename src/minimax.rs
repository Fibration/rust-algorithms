#[cfg(test)]
mod test_minimax {
    use super::*;

    #[test]
    fn test_terminal_state() {
        let node = (1, String::from(""));
        let tree = vec![node.clone()];
        assert_eq!(minimax(node.clone(), tree, true), 1)
    }
}

pub fn minimax(node: (i32, String), tree:Vec<(i32, String)>, maximize: bool) -> i32 {
    node.0
}