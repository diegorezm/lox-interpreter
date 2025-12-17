const std = @import("std");
const interpreter = @import("interpreter");

pub fn main() !void {
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    try interpreter.bufferedPrint();
}
