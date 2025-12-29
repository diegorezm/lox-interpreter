const std = @import("std");
const syntax = @import("syntax");
const debug = std.debug;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var args = try std.process.argsWithAllocator(arena.allocator());
    defer args.deinit();

    _ = args.next();

    const filename = args.next() orelse {
        std.debug.print("USAGE: jlox file.lox\n", .{});
        return;
    };

    const contents = try readFileContent(arena.allocator(), filename);

    std.debug.print(
        "Read {d} bytes from '{s}'\n",
        .{ contents.len, filename },
    );

    var l = syntax.Scanner.init(arena.allocator(), contents);
    try l.scan();

    const tokens = l.getTokens();

    var parser = syntax.Parser.init(arena.allocator(), tokens);
    const expr = try parser.parse();

    var buf: [1024]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&buf);
    try syntax.printExpr(expr, &stdout.interface);

    try stdout.interface.flush();
}

fn readFileContent(alloc: std.mem.Allocator, filename: []const u8) ![]u8 {
    const cwd = std.fs.cwd();

    const stat = cwd.statFile(filename) catch |err| {
        std.debug.print(
            "Error: failed to open '{s}': {s}\n",
            .{ filename, @errorName(err) },
        );
        return err;
    };

    if (stat.kind != .file) {
        std.debug.print("Error: '{s}' is not a regular file\n", .{filename});
        return error.NotAFile;
    }

    const file = cwd.openFile(filename, .{ .mode = .read_only }) catch |err| {
        std.debug.print(
            "Error: failed to open '{s}': {s}\n",
            .{ filename, @errorName(err) },
        );
        return err;
    };

    defer file.close();

    const contents = file.readToEndAlloc(
        alloc,
        stat.size,
    ) catch |err| {
        std.debug.print(
            "Error: failed to read '{s}': {s}\n",
            .{ filename, @errorName(err) },
        );
        return err;
    };
    return contents;
}
