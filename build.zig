const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const syntax_mod = b.addModule("interpreter", .{
        .root_source_file = b.path("src/interpreter.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Bundle shared imports so we can reuse them
    const shared_imports = [_]std.Build.Module.Import{
        .{ .name = "interpreter", .module = syntax_mod },
    };

    // ------------------------------------------------------------
    // Executables
    // ------------------------------------------------------------
    const interpreter_exe = addExecutable(
        b,
        "jlox",
        "src/main.zig",
        target,
        optimize,
        &shared_imports,
    );

    b.installArtifact(interpreter_exe);

    // ------------------------------------------------------------
    // Run step (defaults to interpreter)
    // ------------------------------------------------------------
    const run_cmd = b.addRunArtifact(interpreter_exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run interpreter");
    run_step.dependOn(&run_cmd.step);

    // ------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------

    const exe_tests = b.addTest(.{
        .root_module = interpreter_exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_exe_tests.step);
}

// ------------------------------------------------------------
// Helper function to reduce duplication
// ------------------------------------------------------------
fn addExecutable(
    b: *std.Build,
    name: []const u8,
    root_file: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    imports: []const std.Build.Module.Import,
) *std.Build.Step.Compile {
    return b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(root_file),
            .target = target,
            .optimize = optimize,
            .imports = imports,
        }),
    });
}
