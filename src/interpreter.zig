const std = @import("std");

// ----TYPES ----

pub const TokenType = enum {
    // Single-character tokens.
    LEFT_PAREN,
    RIGHT_PAREN,
    LEFT_BRACE,
    RIGHT_BRACE,
    COMMA,
    DOT,
    MINUS,
    PLUS,
    SEMICOLON,
    SLASH,
    STAR,

    // One or two character tokens.
    BANG,
    BANG_EQUAL,
    EQUAL,
    EQUAL_EQUAL,
    GREATER,
    GREATER_EQUAL,
    LESS,
    LESS_EQUAL,

    // Literals.
    IDENTIFIER,
    STRING,
    NUMBER,

    // Keywords.
    AND,
    CLASS,
    ELSE,
    FALSE,
    FN,
    FOR,
    IF,
    NIL,
    OR,
    PRINT,
    RETURN,
    SUPER,
    THIS,
    TRUE,
    VAR,
    WHILE,

    EOF,
};

pub const Token = struct {
    type: TokenType,
    literal: ?Literal,
    lexeme: []const u8,
    line: usize,

    pub fn init(tokenType: TokenType, literal: ?Literal, lexeme: []const u8, line: usize) Token {
        return .{ .type = tokenType, .literal = literal, .lexeme = lexeme, .line = line };
    }

    pub fn create(allocator: std.mem.Allocator, tokenType: TokenType, literal: ?Literal, lexeme: []const u8, line: usize) !*Token {
        const tokenPtr = try allocator.alloc(Token);
        tokenPtr.* = Token.init(tokenType, literal, lexeme, line);
        return tokenPtr;
    }

    // Turn this token into a json string.
    pub fn toString(self: Token, alloc: std.mem.Allocator) ![]const u8 {
        const fmt = std.json.fmt(self, .{ .whitespace = .indent_2 });

        var writer = std.Io.Writer.Allocating.init(alloc);
        try fmt.format(&writer.writer);

        const json_string = try writer.toOwnedSlice();
        return json_string;
    }
};

pub const Literal = union(enum) {
    number: f64,
    string: []const u8,
    boolean: bool,
    nil: void,

    pub fn format(
        self: Literal,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        switch (self) {
            .number => |n| try writer.print("{d}", .{n}),
            .string => |s| try writer.print("\"{s}\"", .{s}),
            .boolean => |b| try writer.print("{}", .{b}),
            .nil => try writer.print("nil", .{}),
        }
    }
};

pub const LiteralExpr = struct {
    value: Literal,
};

pub const BinaryExpr = struct {
    left: *Expr,
    operator: Token,
    right: *Expr,
};

pub const UnaryExpr = struct {
    operator: Token,
    right: *Expr,
};

pub const LogicalExpr = struct {
    left: *Expr,
    operator: Token,
    right: *Expr,
};

pub const GroupingExpr = struct {
    expression: *Expr,
};

pub const VariableExpr = struct {
    name: Token,
};

pub const AssignExpr = struct {
    name: Token,
    value: *Expr,
};

pub const Value = union(enum) {
    number: f64,
    boolean: bool,
    string: []const u8,
    callable: *LoxCallable,
    nil,
};

pub const Expr = union(enum) { Binary: BinaryExpr, Unary: UnaryExpr, Literal: LiteralExpr, Grouping: GroupingExpr, Variable: VariableExpr, Assign: AssignExpr, Logical: LogicalExpr, Call: CallExpr };

pub const ExprStmt = struct {
    expression: *Expr,
};

pub const CallExpr = struct { callee: *Expr, paren: Token, arguments: ?[]*Expr };

pub const PrintStmt = struct {
    expression: *Expr,
};

const ReturnSignal = struct { value: ?Value };

pub const ReturnStmt = struct { keyword: Token, value: ?*Expr };

pub const VarStmt = struct {
    name: Token,
    initializer: ?*Expr, // `var a;` is valid
};

pub const WhileStmt = struct {
    condition: *Expr,
    body: *Stmt,
};

pub const BlockStmt = struct {
    statements: []const *Stmt,
};

pub const IFStmt = struct {
    condition: *Expr,
    thenBranch: *Stmt,
    elseBranch: ?*Stmt,
};

pub const FunctionStmt = struct {
    name: Token,
    params: []Token,
    body: []const *Stmt,
};

pub const Stmt = union(enum) { expr: ExprStmt, print: PrintStmt, var_decl: VarStmt, block: BlockStmt, if_decl: IFStmt, while_decl: WhileStmt, function_decl: FunctionStmt, return_stmt: ReturnStmt };

pub const Stmts = std.ArrayList(Stmt);

pub const FunctionType = enum { NONE, FUNCTION };

const ParserError = error{ ExpectedExpression, ExpectedRightParen, UnexpectedToken, OutOfMemory, InvalidAssignment, TooManyArguments };

pub const RuntimeError = error{ UndefinedVariable, InvalidOperands, DivisionByZero, OutputError, OutOfMemory, TypeError, TooManyArguments, NotEnoughArguments, InvalidArity, Return, RedeclaredVariable, ReadInOwnInitializer, TopLevelReturn };

pub const UserFunction = struct {
    arity: usize,
    declaration: *const FunctionStmt,
    clousure: *Environment,

    pub fn init(declaration: *const FunctionStmt, clousure: *Environment) UserFunction {
        return .{ .arity = declaration.params.len, .declaration = declaration, .clousure = clousure };
    }

    pub fn call(self: *UserFunction, interpreter: *Interpreter, args: []const Value) RuntimeError!?Value {
        if (args.len != self.arity) return RuntimeError.InvalidArity;
        const env = interpreter.alloc.create(Environment) catch {
            return RuntimeError.OutOfMemory;
        };
        env.* = Environment.init(interpreter.alloc, self.clousure);

        for (0..self.arity) |index| {
            try env.define(self.declaration.params[index].lexeme, args[index]);
        }

        _ = interpreter.execBlock(self.declaration.body, env) catch |err| switch (err) {
            RuntimeError.Return => {
                const rv = interpreter.returnSignal orelse Value{ .nil = {} };
                interpreter.returnSignal = null;
                return rv;
            },
            else => return err,
        };
        return null;
    }
};

pub const NativeFunction = struct {
    arity: usize,
    func: *const fn (*Interpreter, []const Value) RuntimeError!?Value,

    pub fn call(
        self: *NativeFunction,
        interpreter: *Interpreter,
        args: []const Value,
    ) RuntimeError!?Value {
        return try self.func(interpreter, args);
    }
};

// I just learned about this pattern, that is why the other structs don't follow it
// but it is pretty cool
pub const LoxCallable = union(enum) {
    userFn: UserFunction,
    nativeFn: NativeFunction,

    pub fn arity(self: *LoxCallable) usize {
        return switch (self.*) {
            .userFn => self.userFn.arity,
            .nativeFn => self.nativeFn.arity,
        };
    }

    pub fn call(
        self: *LoxCallable,
        interpreter: *Interpreter,
        args: []const Value,
    ) RuntimeError!?Value {
        return switch (self.*) {
            .userFn => |*f| try f.call(interpreter, args),
            .nativeFn => |*nf| try nf.call(interpreter, args),
        };
    }
};
// ----

// ---- Utils ----

pub fn makeBinary(
    allocator: std.mem.Allocator,
    left: *Expr,
    operator: Token,
    right: *Expr,
) !*Expr {
    const node = try allocator.create(Expr);
    node.* = Expr{ .Binary = BinaryExpr{ .left = left, .operator = operator, .right = right } };
    return node;
}

pub fn makeLogical(
    allocator: std.mem.Allocator,
    left: *Expr,
    operator: Token,
    right: *Expr,
) !*Expr {
    const node = try allocator.create(Expr);
    node.* = Expr{ .Logical = LogicalExpr{ .left = left, .operator = operator, .right = right } };
    return node;
}

fn makeAssign(
    alloc: std.mem.Allocator,
    name: Token,
    value: *Expr,
) !*Expr {
    const expr = try alloc.create(Expr);
    expr.* = .{
        .Assign = .{
            .name = name,
            .value = value,
        },
    };
    return expr;
}

pub fn makeCall(allocator: std.mem.Allocator, callee: *Expr, paren: Token, arguments: ?[]*Expr) !*Expr {
    const node = try allocator.create(Expr);
    node.* = Expr{
        .Call = CallExpr{ .callee = callee, .arguments = arguments, .paren = paren },
    };
    return node;
}

pub fn makeLiteral(
    allocator: std.mem.Allocator,
    value: Literal,
) !*Expr {
    const node = try allocator.create(Expr);
    node.* = Expr{
        .Literal = LiteralExpr{
            .value = value,
        },
    };
    return node;
}

pub fn makeUnary(
    allocator: std.mem.Allocator,
    operator: Token,
    right: *Expr,
) !*Expr {
    const node = try allocator.create(Expr);
    node.* = Expr{ .Unary = UnaryExpr{ .operator = operator, .right = right } };
    return node;
}

pub fn makeGrouping(allocator: std.mem.Allocator, expr: *Expr) !*Expr {
    const node = try allocator.create(Expr);
    node.* = Expr{ .Grouping = GroupingExpr{ .expression = expr } };
    return node;
}

fn makeVariable(
    allocator: std.mem.Allocator,
    name: Token,
) !*Expr {
    const node = try allocator.create(Expr);
    node.* = Expr{
        .Variable = VariableExpr{
            .name = name,
        },
    };
    return node;
}

/// Converts a `Value` into an owned UTF-8 string.
/// The returned slice is allocated using `allocator` and must be freed by the caller.
pub fn valueToString(
    allocator: std.mem.Allocator,
    value: Value,
) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();

    try printValue(value, list.writer());
    return list.toOwnedSlice();
}

/// Writes a `Value` to an arbitrary `std.io.Writer`.
pub fn printValue(value: Value, writer: *std.io.Writer) !void {
    switch (value) {
        .number => |n| try writer.print("{d}", .{n}),
        .boolean => |b| try writer.print("{}", .{b}),
        .string => |s| try writer.print("{s}", .{s}),
        .callable => |s| {
            switch (s.*) {
                .nativeFn => |native| {
                    try writer.print("<fn @{x}>", .{@intFromPtr(native.func)});
                },
                .userFn => |user| {
                    try writer.print("<fn {s} >", .{user.declaration.name.lexeme});
                },
            }
        },
        .nil => try writer.writeAll("nil"),
    }
}

fn escapeString(alloc: std.mem.Allocator, s: []const u8) ![]u8 {
    var buf = try alloc.alloc(u8, s.len);
    var len: usize = 0;

    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\' and i + 1 < s.len) {
            i += 1;
            switch (s[i]) {
                'n' => buf[len] = '\n',
                't' => buf[len] = '\t',
                '\\' => buf[len] = '\\',
                '"' => buf[len] = '"',
                else => buf[len] = s[i], // unknown escape → raw
            }
        } else {
            buf[len] = s[i];
        }
        len += 1;
    }

    return buf[0..len];
}
pub fn printExpr(expr: *const Expr, writer: *std.io.Writer) !void {
    switch (expr.*) {
        .Literal => |lit| {
            try printLiteral(lit.value, writer);
        },
        .Grouping => |g| {
            try writer.writeAll("(group ");
            try printExpr(g.expression, writer);
            try writer.writeByte(')');
        },
        .Unary => |u| {
            try writer.print("({s} ", .{u.operator.lexeme});
            try printExpr(u.right, writer);
            try writer.writeByte(')');
        },
        .Binary => |b| {
            try writer.print("({s} ", .{b.operator.lexeme});
            try printExpr(b.left, writer);
            try writer.writeByte(' ');
            try printExpr(b.right, writer);
            try writer.writeByte(')');
        },
    }
}

pub fn printLiteral(lit: Literal, writer: *std.io.Writer) !void {
    switch (lit) {
        .number => |n| try writer.print("{d}", .{n}),
        .string => |s| try writer.print("\"{s}\"", .{s}),
        .boolean => |b| try writer.print("{}", .{b}),
        .nil => try writer.writeAll("nil"),
    }
}

const keywords = std.StaticStringMap(TokenType).initComptime(.{
    .{ "and", .AND },
    .{ "class", .CLASS },
    .{ "else", .ELSE },
    .{ "false", .FALSE },
    .{ "FALSE", .FALSE },
    .{ "for", .FOR },
    .{ "fn", .FN },
    .{ "if", .IF },
    .{ "nil", .NIL },
    .{ "or", .OR },
    .{ "print", .PRINT },
    .{ "return", .RETURN },
    .{ "super", .SUPER },
    .{ "this", .THIS },
    .{ "true", .TRUE },
    .{ "TRUE", .TRUE },
    .{ "var", .VAR },
    .{ "while", .WHILE },
});

/// Gets a random `u8` and tries to turn it into a `TokenType`.
fn keywordType(text: []const u8) TokenType {
    return keywords.get(text) orelse .IDENTIFIER;
}

// ----

// ---- Scanner ----

//// This struct should read
pub const Scanner = struct {
    tokens: std.ArrayList(Token) = .empty,
    source: []const u8,
    alloc: std.mem.Allocator,
    start: usize,
    current: usize,
    line: usize,

    pub fn init(alloc: std.mem.Allocator, source: []const u8) Scanner {
        return .{ .tokens = .empty, .alloc = alloc, .source = source, .start = 0, .current = 0, .line = 0 };
    }

    pub fn getTokens(self: *Scanner) std.ArrayList(Token) {
        return self.tokens;
    }

    pub fn scan(self: *Scanner) !void {
        while (!self.isAtEnd()) {
            self.start = self.current;
            try self.scanToken();
        }

        try self.addToken(.EOF, null);
    }

    fn scanToken(self: *Scanner) !void {
        const c = self.advance();

        switch (c) {
            '(' => try self.addToken(.LEFT_PAREN, null),
            ')' => try self.addToken(.RIGHT_PAREN, null),
            '{' => try self.addToken(.LEFT_BRACE, null),
            '}' => try self.addToken(.RIGHT_BRACE, null),
            ',' => try self.addToken(.COMMA, null),
            '.' => try self.addToken(.DOT, null),
            '-' => try self.addToken(.MINUS, null),
            '+' => try self.addToken(.PLUS, null),
            ';' => try self.addToken(.SEMICOLON, null),
            '*' => try self.addToken(.STAR, null),

            '!' => try self.addToken(
                if (self.match('=')) .BANG_EQUAL else .BANG,
                null,
            ),

            '=' => try self.addToken(
                if (self.match('=')) .EQUAL_EQUAL else .EQUAL,
                null,
            ),

            '<' => try self.addToken(
                if (self.match('=')) .LESS_EQUAL else .LESS,
                null,
            ),

            '>' => try self.addToken(
                if (self.match('=')) .GREATER_EQUAL else .GREATER,
                null,
            ),

            '"' => try self.scanString(),

            ' ', '\r', '\t' => {},
            '\n' => self.line += 1,

            else => {
                if (std.ascii.isDigit(c)) {
                    try self.scanNumber();
                } else if (std.ascii.isAlphabetic(c) or c == '_') {
                    try self.scanIdentifier();
                } else {
                    return self.errorAtCurrent("Unexpected character");
                }
            },
        }
    }

    fn scanNumber(self: *Scanner) !void {
        while (std.ascii.isDigit(self.peek() orelse 0)) {
            _ = self.advance();
        }

        if (self.peek() == '.' and std.ascii.isDigit(self.peekNext() orelse 0)) {
            _ = self.advance();

            while (std.ascii.isDigit(self.peek() orelse 0)) {
                _ = self.advance();
            }
        }

        const text = self.source[self.start..self.current];
        const value = try std.fmt.parseFloat(f64, text);

        try self.addToken(.NUMBER, Literal{ .number = value });
    }

    fn scanString(
        self: *Scanner,
    ) !void {
        while (self.peek()) |c| {
            if (c == '"') break;

            if (c == '\n') {
                self.line += 1;
            }

            _ = self.advance();
        }

        if (self.isAtEnd()) {
            return self.errorAtCurrent("Unterminated string literal. Expected '\"'.");
        }

        _ = self.advance();

        const value = self.source[self.start + 1 .. self.current - 1];

        const lexeme = try self.alloc.dupe(
            u8,
            self.source[self.start..self.current],
        );

        try self.tokens.append(
            self.alloc,
            Token.init(
                .STRING,
                Literal{ .string = value },
                lexeme,
                self.line,
            ),
        );
    }

    fn scanIdentifier(self: *Scanner) !void {
        while (self.peek()) |c| {
            if (!std.ascii.isAlphanumeric(c) and c != '_') break;
            _ = self.advance();
        }

        const text = self.source[self.start..self.current];
        const token_type = keywordType(text);

        const literal: ?Literal = switch (token_type) {
            .TRUE => .{ .boolean = true },
            .FALSE => .{ .boolean = false },
            .NIL => .{ .nil = {} },
            else => null,
        };

        try self.addToken(token_type, literal);
    }

    fn addToken(
        self: *Scanner,
        token_type: TokenType,
        literal: ?Literal,
    ) !void {
        const text = self.source[self.start..self.current];
        const lexeme = try self.alloc.dupe(u8, text);

        try self.tokens.append(
            self.alloc,
            Token.init(token_type, literal, lexeme, self.line),
        );
    }

    fn errorAtCurrent(self: *Scanner, message: []const u8) error{LexError} {
        std.log.err(
            "[line {}] Error at '{}': {s}\n",
            .{ self.line, self.source[self.current - 1], message },
        );
        return error.LexError;
    }

    fn peekNext(self: *Scanner) ?u8 {
        if (self.current + 1 >= self.source.len) return null;
        return self.source[self.current + 1];
    }

    fn isAtEnd(self: *Scanner) bool {
        return self.current >= self.source.len;
    }

    fn advance(self: *Scanner) u8 {
        const c = self.source[self.current];
        self.current += 1;
        return c;
    }

    fn peek(self: *Scanner) ?u8 {
        if (self.isAtEnd()) return null;
        return self.source[self.current];
    }

    fn match(self: *Scanner, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;
        self.current += 1;
        return true;
    }
};

// ----

/// This struct parses the token stream produced by `Scanner`.
/// See: Scanner
pub const Parser = struct {
    current: usize = 0,
    tokens: std.ArrayList(Token),
    alloc: std.mem.Allocator,

    pub fn init(
        alloc: std.mem.Allocator,
        tokens: std.ArrayList(Token),
    ) Parser {
        return .{ .alloc = alloc, .tokens = tokens };
    }

    pub fn parse(self: *Parser) ![]const *Stmt {
        var stmts: std.ArrayList(*Stmt) = .empty;

        while (!self.isAtEnd()) {
            if (self.declaration()) |stmt| {
                try stmts.append(self.alloc, stmt);
            }
        }

        const items = stmts.items;
        return items;
    }

    fn declaration(self: *Parser) ?*Stmt {
        if (self.match(&[_]TokenType{.FN})) {
            return self.function("function") catch {
                self.sync();
                return null;
            };
        }

        if (self.match(&[_]TokenType{.VAR})) {
            return self.varDeclaration() catch {
                self.sync();
                return null;
            };
        }

        return self.statement() catch {
            self.sync();
            return null;
        };
    }

    fn statement(self: *Parser) ParserError!*Stmt {
        if (self.match(&[_]TokenType{TokenType.PRINT})) return try self.printStatement();
        if (self.match(&[_]TokenType{TokenType.FOR})) return try self.forStatement();
        if (self.match(&[_]TokenType{TokenType.IF})) return try self.ifStatement();
        if (self.match(&[_]TokenType{TokenType.RETURN})) return try self.returnStatement();
        if (self.match(&[_]TokenType{TokenType.WHILE})) return try self.whileStatement();
        if (self.match(&[_]TokenType{TokenType.LEFT_BRACE})) {
            const b = try self.block();

            const stmt = self.alloc.create(Stmt) catch {
                return ParserError.OutOfMemory;
            };

            stmt.* = Stmt{ .block = BlockStmt{ .statements = b } };
            return stmt;
        }
        return try self.expressionStatement();
    }

    fn forStatement(self: *Parser) ParserError!*Stmt {
        _ = try self.consume(.LEFT_PAREN, "Expect '(' after 'for'.");

        var initializer: ?*Stmt = null;
        if (self.match(&[_]TokenType{.SEMICOLON})) {
            initializer = null;
        } else if (self.match(&[_]TokenType{.VAR})) {
            initializer = try self.varDeclaration();
        } else {
            initializer = try self.expressionStatement();
        }

        var condition: ?*Expr = null;
        if (!self.check(.SEMICOLON)) {
            condition = try self.expression();
        }
        _ = try self.consume(.SEMICOLON, "Expect ';' after loop condition.");

        var increment: ?*Expr = null;
        if (!self.check(.RIGHT_PAREN)) {
            increment = try self.expression();
        }
        _ = try self.consume(.RIGHT_PAREN, "Expect ')' after for clauses.");

        var body = try self.statement(); // body: *Stmt

        if (increment) |inc| {
            var stmts: std.ArrayList(*Stmt) = .empty;
            errdefer stmts.deinit(self.alloc);

            try stmts.append(self.alloc, body);

            const incStmt = try self.alloc.create(Stmt);
            incStmt.* = Stmt{
                .expr = ExprStmt{ .expression = inc },
            };

            try stmts.append(self.alloc, incStmt);
            const blockStmt = try self.alloc.create(Stmt);

            blockStmt.* = Stmt{
                .block = BlockStmt{ .statements = stmts.items },
            };
            body = blockStmt;
        }

        if (condition == null) {
            condition = try makeLiteral(self.alloc, .{ .boolean = true });
        }

        const whileStmt = try self.alloc.create(Stmt);
        whileStmt.* = Stmt{
            .while_decl = WhileStmt{
                .condition = condition.?,
                .body = body,
            },
        };
        body = whileStmt;

        if (initializer) |i| {
            var stmts: std.ArrayList(*Stmt) = .empty;
            errdefer stmts.deinit(self.alloc);

            try stmts.append(self.alloc, i);
            try stmts.append(self.alloc, body);

            const blockStmt = try self.alloc.create(Stmt);
            blockStmt.* = Stmt{
                .block = BlockStmt{ .statements = stmts.items },
            };
            body = blockStmt;
        }

        return body;
    }

    fn ifStatement(self: *Parser) ParserError!*Stmt {
        _ = try self.consume(.LEFT_PAREN, "Expect '(' after 'if'.");

        const condition = try self.expression();

        _ = try self.consume(.RIGHT_PAREN, "Expect ')' after if condition.");

        const thenBranch = try self.statement();
        var elseBranch: ?*Stmt = null;

        if (self.match(&[_]TokenType{.ELSE})) {
            elseBranch = try self.statement();
        }

        const stmt = try self.alloc.create(Stmt);

        stmt.* = .{ .if_decl = IFStmt{ .condition = condition, .thenBranch = thenBranch, .elseBranch = elseBranch } };

        return stmt;
    }
    fn whileStatement(self: *Parser) ParserError!*Stmt {
        _ = try self.consume(.LEFT_PAREN, "Expect '(' after 'while'.");
        const condition = try self.expression();
        _ = try self.consume(.RIGHT_PAREN, "Expect ')' after condition.");
        const body = try self.statement();

        const stmt = try self.alloc.create(Stmt);
        stmt.* = .{ .while_decl = WhileStmt{ .condition = condition, .body = body } };

        return stmt;
    }

    fn varDeclaration(self: *Parser) ParserError!*Stmt {
        const name = try self.consume(TokenType.IDENTIFIER, "Expect variable name.");

        var initializer: ?*Expr = null;
        const tokenTypes = [_]TokenType{TokenType.EQUAL};

        if (self.match(&tokenTypes)) {
            initializer = try self.expression();
        }

        _ = try self.consume(TokenType.SEMICOLON, "Expect ';' after value.");

        const stmt = try self.alloc.create(Stmt);
        stmt.* = .{ .var_decl = VarStmt{ .initializer = initializer, .name = name } };

        return stmt;
    }

    fn printStatement(self: *Parser) ParserError!*Stmt {
        const value = try self.expression();
        _ = try self.consume(TokenType.SEMICOLON, "Expect ';' after value.");
        const stmt = try self.alloc.create(Stmt);
        stmt.* = .{ .print = PrintStmt{ .expression = value } };

        return stmt;
    }

    fn returnStatement(self: *Parser) ParserError!*Stmt {
        const keyword = self.previous();
        var value: ?*Expr = null;

        if (!self.check(.SEMICOLON)) {
            value = try self.expression();
        }
        _ = try self.consume(TokenType.SEMICOLON, "Expect ';' after value.");

        const stmt = try self.alloc.create(Stmt);
        stmt.* = .{ .return_stmt = ReturnStmt{ .keyword = keyword, .value = value } };

        return stmt;
    }

    fn expressionStatement(self: *Parser) ParserError!*Stmt {
        const value = try self.expression();
        _ = try self.consume(TokenType.SEMICOLON, "Expect ';' after value.");
        const stmt = try self.alloc.create(Stmt);
        stmt.* = .{
            .expr = ExprStmt{
                .expression = value,
            },
        };

        return stmt;
    }

    fn function(self: *Parser, fnKind: []const u8) ParserError!*Stmt {
        var buf: [64]u8 = undefined;

        const msg_name = std.fmt.bufPrint(&buf, "Expect {s} name.", .{fnKind}) catch {
            return ParserError.OutOfMemory;
        };

        const name = try self.consume(.IDENTIFIER, msg_name);

        const msg_paren = std.fmt.bufPrint(&buf, "Expect '(' after {s} name.", .{fnKind}) catch {
            return ParserError.OutOfMemory;
        };
        _ = try self.consume(.LEFT_PAREN, msg_paren);

        var parameters = std.ArrayList(Token).initCapacity(self.alloc, 128) catch {
            return ParserError.OutOfMemory;
        };

        if (!self.check(.RIGHT_PAREN)) {
            while (true) {
                if (parameters.items.len >= 128) return ParserError.TooManyArguments;
                const param = try self.consume(.IDENTIFIER, "Expect parameter name.");
                parameters.append(self.alloc, param) catch {
                    return ParserError.OutOfMemory;
                };
                if (!self.match(&[_]TokenType{.COMMA})) break;
            }
        }

        _ = try self.consume(.RIGHT_PAREN, "Expect ')' after parameters.");

        const msg_body = std.fmt.bufPrint(&buf, "Expect '{{' before {s} body.", .{fnKind}) catch {
            return ParserError.OutOfMemory;
        };
        _ = try self.consume(.LEFT_BRACE, msg_body);

        const body = try self.block();

        const stmt = try self.alloc.create(Stmt);

        stmt.* = Stmt{
            .function_decl = FunctionStmt{
                .name = name,
                .params = parameters.items,
                .body = body,
            },
        };

        return stmt;
    }

    fn expression(self: *Parser) ParserError!*Expr {
        return try self.assignment();
    }

    fn assignment(self: *Parser) ParserError!*Expr {
        const expr = try self.orExpr();

        const tokenTypes = [_]TokenType{TokenType.EQUAL};
        if (self.match(&tokenTypes)) {
            _ = self.previous();
            const value = try self.assignment();

            switch (expr.*) {
                .Variable => |variable| {
                    return try makeAssign(self.alloc, variable.name, value);
                },
                else => {
                    return ParserError.InvalidAssignment;
                },
            }
        }
        return expr;
    }

    fn orExpr(self: *Parser) !*Expr {
        var expr = try self.andExpr();

        const tokenTypes = [_]TokenType{TokenType.OR};

        while (self.match(&tokenTypes)) {
            const operator = self.previous();
            const right = try self.andExpr();
            expr = try makeLogical(self.alloc, expr, operator, right);
        }
        return expr;
    }

    fn andExpr(self: *Parser) !*Expr {
        var expr = try self.equality();

        const tokenTypes = [_]TokenType{TokenType.AND};

        while (self.match(&tokenTypes)) {
            const operator = self.previous();
            const right = try self.equality();
            expr = try makeLogical(self.alloc, expr, operator, right);
        }
        return expr;
    }

    fn equality(self: *Parser) !*Expr {
        var expr = try self.comparison();
        const tokenTypes = [_]TokenType{ TokenType.BANG_EQUAL, TokenType.EQUAL_EQUAL };

        while (self.match(&tokenTypes)) {
            const operator = self.previous();
            const right = try self.comparison();
            expr = try makeBinary(self.alloc, expr, operator, right);
        }

        return expr;
    }

    fn comparison(self: *Parser) !*Expr {
        var expr = try self.term();

        const tokenTypes = [_]TokenType{ TokenType.GREATER, TokenType.GREATER_EQUAL, TokenType.LESS_EQUAL, TokenType.LESS };

        while (self.match(&tokenTypes)) {
            const operator = self.previous();
            const right = try self.term();
            expr = try makeBinary(self.alloc, expr, operator, right);
        }

        return expr;
    }

    fn term(self: *Parser) ParserError!*Expr {
        var expr = try self.factor();

        const tokenTypes = [_]TokenType{ TokenType.PLUS, TokenType.MINUS };

        while (self.match(&tokenTypes)) {
            const operator = self.previous();
            const right = try self.factor();
            expr = try makeBinary(self.alloc, expr, operator, right);
        }

        return expr;
    }

    fn factor(self: *Parser) ParserError!*Expr {
        var expr = try self.unary();

        const tokenTypes = [_]TokenType{ TokenType.STAR, TokenType.SLASH };

        while (self.match(&tokenTypes)) {
            const operator = self.previous();
            const right = try self.term();
            expr = try makeBinary(self.alloc, expr, operator, right);
        }

        return expr;
    }

    fn unary(self: *Parser) ParserError!*Expr {
        const tokenTypes = [_]TokenType{ TokenType.BANG, TokenType.MINUS };

        while (self.match(&tokenTypes)) {
            const operator = self.previous();
            const right = try self.unary();
            return try makeUnary(self.alloc, operator, right);
        }

        return try self.call();
    }

    fn call(self: *Parser) ParserError!*Expr {
        var expr = try self.primary();
        while (true) {
            if (self.match(&[_]TokenType{.LEFT_PAREN})) {
                expr = try self.finishCall(expr);
            } else {
                break;
            }
        }
        return expr;
    }

    fn finishCall(self: *Parser, callee: *Expr) ParserError!*Expr {
        var args = std.ArrayList(*Expr).initCapacity(self.alloc, 128) catch {
            return ParserError.OutOfMemory;
        };

        if (!self.check(.RIGHT_PAREN)) {
            // check if there is a comma, if there is none then it means the function call
            // has no more args.
            const first = try self.expression(); // I have to do this because this language does not have do whiles (i think)
            args.append(self.alloc, first) catch {
                return ParserError.OutOfMemory;
            };
            while (self.match(&[_]TokenType{.COMMA})) {
                if (args.items.len >= 127) {
                    const p = self.peek();
                    reportParseError(p.?, "Too many arguments.");
                }

                const expr = try self.expression();
                args.append(self.alloc, expr) catch {
                    return ParserError.OutOfMemory;
                };
            }
        }

        const paren = try self.consume(.RIGHT_PAREN, "Expect ')' after arguments.");
        return try makeCall(self.alloc, callee, paren, args.items);
    }

    fn primary(self: *Parser) ParserError!*Expr {
        if (self.match(&[_]TokenType{.FALSE})) {
            return try makeLiteral(self.alloc, Literal{ .boolean = false });
        }

        if (self.match(&[_]TokenType{.TRUE})) {
            return try makeLiteral(self.alloc, Literal{ .boolean = true });
        }

        if (self.match(&[_]TokenType{.NIL})) {
            return try makeLiteral(self.alloc, Literal{ .nil = {} });
        }

        if (self.match(&[_]TokenType{ .NUMBER, .STRING })) {
            const lit = self.previous().literal.?;
            return try makeLiteral(self.alloc, lit);
        }

        if (self.match(&[_]TokenType{.IDENTIFIER})) {
            const token = self.previous();
            return try makeVariable(self.alloc, token);
        }

        if (self.match(&[_]TokenType{.LEFT_PAREN})) {
            const expr = try self.expression();
            _ = try self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.");
            return try makeGrouping(self.alloc, expr);
        }

        if (self.isAtEnd()) {
            return ParserError.ExpectedExpression;
        }

        // const p = self.peek().?;
        // reportParseError(p, "Expected expression.");
        return ParserError.ExpectedExpression;
    }

    /// Synchronizes the parser after an error.
    ///
    /// When a parse error occurs, the parser may be in an invalid state.
    /// This function discards tokens until it finds a point where parsing
    /// can safely resume, preventing multiple misleading error messages
    /// caused by a single mistake.
    fn sync(self: *Parser) void {
        _ = self.advance();

        while (!self.isAtEnd()) {
            if (self.previous().type == .SEMICOLON) {
                return;
            }

            switch (self.peek().?.type) {
                .CLASS,
                .FN,
                .VAR,
                .FOR,
                .IF,
                .WHILE,
                .PRINT,
                .RETURN,
                => return,

                else => {},
            }

            _ = self.advance();
        }
    }

    fn block(self: *Parser) ParserError![]const *Stmt {
        var stmts: std.ArrayList(*Stmt) = .empty;

        while (!self.check(TokenType.RIGHT_BRACE) and !self.isAtEnd()) {
            if (self.declaration()) |stmtPtr| {
                stmts.append(self.alloc, stmtPtr) catch {
                    return ParserError.OutOfMemory;
                };
            }
        }
        _ = try self.consume(TokenType.RIGHT_BRACE, "Expect '}' after block.");
        return stmts.items;
    }

    fn match(self: *Parser, tokenTypes: []const TokenType) bool {
        for (tokenTypes) |t| {
            if (self.check(t)) {
                _ = self.advance();
                return true;
            }
        }
        return false;
    }

    fn consume(self: *Parser, t: TokenType, message: []const u8) ParserError!Token {
        if (self.check(t)) return self.advance();

        const p = self.peek();
        if (p == null) return ParserError.ExpectedExpression;

        reportParseError(p.?, message);
        return ParserError.ExpectedExpression;
    }

    fn peek(self: *Parser) ?Token {
        if (self.isAtEnd()) return null;
        return self.tokens.items[self.current];
    }

    fn isAtEnd(self: *Parser) bool {
        return self.current >= self.tokens.items.len;
    }

    fn advance(self: *Parser) Token {
        if (!self.isAtEnd()) self.current += 1;
        return self.previous();
    }

    fn previous(self: *Parser) Token {
        return self.tokens.items[self.current - 1];
    }

    fn check(self: *Parser, t: TokenType) bool {
        if (self.isAtEnd()) return false;

        const p = self.peek();
        if (p == null) return false;

        return p.?.type == t;
    }

    fn reportParseError(token: Token, message: []const u8) void {
        if (token.type == .EOF) {
            std.log.err(
                "[line {}] Error at end: {s}\n",
                .{ token.line, message },
            );
        } else {
            std.log.err(
                "[line {}] Error at '{s}': {s}\n",
                .{ token.line, token.lexeme, message },
            );
        }
    }
};

/// Lexical environment that stores variables.
/// This structure is used for both the global scope and block scopes.
// Each environment may have an enclosing pointer, allowing variable
/// lookup to walk outward through nested scopes.
///
/// For example:
///
///   ```zig
///   var a = 12;
///
///   fn abc() {
///     return a + 9;
///   }
///   ```
///
/// This works because the environment created for the function `abc`
/// has access to its own local environment as well as the enclosing
/// global environment where `a` is defined.
pub const Environment = struct {
    values: std.StringHashMap(Value),
    alloc: std.mem.Allocator,
    enclosing: ?*Environment,

    /// Initializes an environment with a allocator and a optional pointer to a enclosing environment.
    pub fn init(alloc: std.mem.Allocator, enclosing: ?*Environment) Environment {
        return .{
            .values = std.StringHashMap(Value).init(alloc),
            .alloc = alloc,
            .enclosing = enclosing,
        };
    }

    /// Defines a entry inside of the environment hahsmap.
    pub fn define(
        self: *Environment,
        name: []const u8,
        value: Value,
    ) RuntimeError!void {
        self.values.put(name, value) catch {
            return RuntimeError.OutOfMemory;
        };
    }

    pub fn getAt(self: *Environment, distance: usize, name: []const u8) RuntimeError!Value {
        if (self.ancestors(distance)) |env| {
            if (env.values.get(name)) |v| {
                return v;
            }
            return RuntimeError.UndefinedVariable;
        }
        return RuntimeError.UndefinedVariable;
    }

    pub fn assignAt(self: *Environment, distance: usize, name: []const u8, value: Value) RuntimeError!void {
        if (self.ancestors(distance)) |env| {
            try env.values.put(name, value);
        } else {
            return RuntimeError.UndefinedVariable;
        }
    }

    fn ancestors(self: *Environment, distance: usize) ?*Environment {
        var enviroment: ?*Environment = self;

        for (0..distance) |_| {
            if (enviroment) |env| {
                enviroment = env.enclosing;
            }
        }

        return enviroment;
    }

    // Assigns a variable. This is different from `define` because here we are not creating any new
    // entries in the hashmap, only modifying entries that already exists.
    pub fn assign(
        self: *Environment,
        name: []const u8,
        value: Value,
    ) RuntimeError!void {
        if (self.values.contains(name)) {
            self.values.put(name, value) catch {
                return RuntimeError.OutOfMemory;
            };
            return;
        }

        if (self.enclosing) |env| {
            return env.assign(name, value);
        }

        return RuntimeError.UndefinedVariable;
    }

    /// Gets a token from the environment.
    pub fn get(
        self: *Environment,
        name: []const u8,
    ) RuntimeError!Value {
        if (self.values.get(name)) |value| {
            return value;
        }

        if (self.enclosing) |env| {
            return env.get(name);
        }

        return RuntimeError.UndefinedVariable;
    }
};

// Resolves variable references to their static scope declarations.
// Enables functions to share a single environment instead of creating new ones each call.
pub const Resolver = struct {
    alloc: std.mem.Allocator,
    interpreter: *Interpreter,
    scopes: std.ArrayList(std.StringHashMap(bool)),
    currentFunction: FunctionType = FunctionType.NONE,

    pub fn init(alloc: std.mem.Allocator, interpreter: *Interpreter) !Resolver {
        return .{ .alloc = alloc, .interpreter = interpreter, .scopes = try std.ArrayList(std.StringHashMap(bool)).initCapacity(alloc, 1048), .currentFunction = FunctionType.NONE };
    }

    pub fn deinit(self: *Resolver) void {
        for (self.scopes.items) |*scope| scope.deinit();
        self.scopes.deinit(self.alloc);
    }

    fn beginScope(self: *Resolver) RuntimeError!void {
        const scope = std.StringHashMap(bool).init(self.alloc);
        try self.scopes.append(self.alloc, scope);
    }

    fn endScope(self: *Resolver) void {
        if (self.scopes.pop()) |scope| {
            var s = scope;
            s.deinit();
        }
    }

    fn isEmpty(self: *Resolver) bool {
        return self.scopes.items.len == 0;
    }

    fn len(self: *Resolver) usize {
        return self.scopes.items.len;
    }

    fn peekMutable(self: *Resolver) ?*std.StringHashMap(bool) {
        if (self.isEmpty()) return null;
        return &self.scopes.items[self.scopes.items.len - 1];
    }

    fn declare(self: *Resolver, name: Token) RuntimeError!void {
        if (self.isEmpty()) return;

        var scope = self.peekMutable().?;
        if (scope.contains(name.lexeme)) {
            return RuntimeError.RedeclaredVariable;
        }
        try scope.put(name.lexeme, false);
    }

    fn define(self: *Resolver, name: Token) void {
        if (self.isEmpty()) return;
        var scope = self.peekMutable().?;
        _ = scope.put(name.lexeme, true) catch {};
    }

    pub fn resolve(self: *Resolver, stmts: []const *Stmt) RuntimeError!void {
        for (stmts) |stmt| {
            try self.resolveStmt(stmt);
        }
    }

    fn resolveStmt(self: *Resolver, stmt: *Stmt) RuntimeError!void {
        switch (stmt.*) {
            .block => |b| {
                try self.beginScope();
                try self.resolve(b.statements);
                self.endScope();
            },
            .var_decl => |v| try self.visitVarStmt(v),
            .function_decl => |f| try self.visitFunctionStmt(f),
            .expr => |e| try self.visitExpressionStmt(e),
            .if_decl => |i| try self.visitIfStmt(i),
            .while_decl => |w| try self.visitWhileStmt(w),
            .print => |p| try self.visitPrintStmt(p),
            .return_stmt => |r| try self.visitReturnStmt(r),
        }
    }

    fn resolveExpr(self: *Resolver, expr: *Expr) RuntimeError!void {
        switch (expr.*) {
            .Assign => |a| try self.visitAssignExpr(expr, a),
            .Variable => |v| try self.visitVariableExpr(expr, v),
            .Binary => |b| try self.visitBinaryExpr(b),
            .Call => |c| try self.visitCallExpr(c),
            .Grouping => |g| try self.visitGroupingExpr(g),
            .Logical => |l| try self.visitLogicalExpr(l),
            .Unary => |u| try self.resolveExpr(u.right),
            .Literal => |l| try self.visitLiteralExpr(l),
        }
    }

    fn resolveLocal(self: *Resolver, expr: *Expr, name: Token) void {
        const scopes_len = self.scopes.items.len;
        var i: usize = scopes_len;
        while (i > 0) : (i -= 1) {
            const scope_ref = &self.scopes.items[i - 1];
            if (scope_ref.contains(name.lexeme)) {
                const distance = scopes_len - i;
                self.interpreter.resolve(expr, distance);
                return;
            }
        }
    }

    fn resolveFunction(self: *Resolver, function: FunctionStmt, functionType: FunctionType) RuntimeError!void {
        const enclosingFunction = self.currentFunction;
        self.currentFunction = functionType;

        try self.beginScope();
        for (function.params) |param| {
            try self.declare(param);
            self.define(param);
        }
        try self.resolve(function.body);
        self.endScope();

        self.currentFunction = enclosingFunction;
    }

    fn visitVarStmt(self: *Resolver, stmt: VarStmt) RuntimeError!void {
        try self.declare(stmt.name);
        if (stmt.initializer) |i| try self.resolveExpr(i);
        self.define(stmt.name);
    }

    fn visitVariableExpr(self: *Resolver, expr_ptr: *Expr, expr: VariableExpr) RuntimeError!void {
        if (!self.isEmpty()) {
            const scope = self.peekMutable().?;
            if (scope.get(expr.name.lexeme)) |initialized| {
                if (!initialized) {
                    return RuntimeError.ReadInOwnInitializer;
                }
            }
        }
        self.resolveLocal(expr_ptr, expr.name);
    }

    fn visitAssignExpr(self: *Resolver, expr_ptr: *Expr, expr: AssignExpr) RuntimeError!void {
        try self.resolveExpr(expr.value);
        self.resolveLocal(expr_ptr, expr.name);
    }

    fn visitExpressionStmt(self: *Resolver, stmt: ExprStmt) RuntimeError!void {
        try self.resolveExpr(stmt.expression);
    }

    fn visitIfStmt(self: *Resolver, stmt: IFStmt) RuntimeError!void {
        try self.resolveExpr(stmt.condition);
        try self.resolveStmt(stmt.thenBranch);
        if (stmt.elseBranch) |else_branch| try self.resolveStmt(else_branch);
    }

    fn visitWhileStmt(self: *Resolver, stmt: WhileStmt) RuntimeError!void {
        try self.resolveExpr(stmt.condition);
        try self.resolveStmt(stmt.body);
    }

    fn visitPrintStmt(self: *Resolver, stmt: PrintStmt) RuntimeError!void {
        try self.resolveExpr(stmt.expression);
    }

    fn visitReturnStmt(self: *Resolver, stmt: ReturnStmt) RuntimeError!void {
        if (self.currentFunction == .NONE) {
            return RuntimeError.TopLevelReturn;
        }
        if (stmt.value) |expr| try self.resolveExpr(expr);
    }

    fn visitBinaryExpr(self: *Resolver, expr: BinaryExpr) RuntimeError!void {
        try self.resolveExpr(expr.left);
        try self.resolveExpr(expr.right);
    }

    fn visitCallExpr(self: *Resolver, expr: CallExpr) RuntimeError!void {
        try self.resolveExpr(expr.callee);
        if (expr.arguments) |args| for (args) |arg| try self.resolveExpr(arg);
    }

    fn visitGroupingExpr(self: *Resolver, expr: GroupingExpr) RuntimeError!void {
        try self.resolveExpr(expr.expression);
    }

    fn visitLogicalExpr(self: *Resolver, expr: LogicalExpr) RuntimeError!void {
        try self.resolveExpr(expr.left);
        try self.resolveExpr(expr.right);
    }

    fn visitLiteralExpr(_: *Resolver, _: LiteralExpr) RuntimeError!void {}

    fn visitFunctionStmt(self: *Resolver, stmt: FunctionStmt) RuntimeError!void {
        try self.declare(stmt.name);
        self.define(stmt.name);
        try self.resolveFunction(stmt, FunctionType.FUNCTION);
    }
};

/// Executes the AST produced by the `Parser` and evaluates expressions.
pub const Interpreter = struct {
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    globalEnv: *Environment,
    env: *Environment,
    returnSignal: ?Value,
    locals: std.AutoHashMap(*Expr, usize),

    pub fn init(alloc: std.mem.Allocator, writer: *std.io.Writer) !Interpreter {
        const globalEnv = try alloc.create(Environment);
        globalEnv.* = Environment.init(alloc, null);

        const clockNativeFn = NativeFunction{ .arity = 0, .func = &clock_fn };
        const callable = alloc.create(LoxCallable) catch {
            return RuntimeError.OutOfMemory;
        };

        callable.* = LoxCallable{ .nativeFn = clockNativeFn };

        try globalEnv.define("clock", Value{ .callable = callable });

        const env = globalEnv;
        const locals: std.AutoHashMap(*Expr, usize) = .init(alloc);

        return .{ .alloc = alloc, .writer = writer, .env = env, .globalEnv = globalEnv, .returnSignal = null, .locals = locals };
    }

    pub fn deinit(self: *Interpreter) void {
        self.locals.deinit();
    }

    pub fn interpret(self: *Interpreter, stmts: []const *Stmt) void {
        for (stmts) |stmtPtr| {
            self.exec(stmtPtr) catch |err| {
                self.runtimeError(err, null, null);
                return;
            };
        }
    }

    fn eval(self: *Interpreter, expr: *Expr) RuntimeError!Value {
        return switch (expr.*) {
            .Literal => |lit| self.visitLiteralExpr(lit),
            .Grouping => |grp| try self.eval(grp.expression),
            .Binary => |binary| try self.visitBinaryExpr(binary),
            .Unary => |unary| try self.visitUnaryExpr(unary),
            .Variable => |variable| try self.visitVariableExpr(expr, variable),
            .Assign => |assign| try self.visitAssignExpr(expr, assign),
            .Logical => |logical| try self.visitLogicalExpr(logical),
            .Call => |call| try self.visitCallExpr(call),
        };
    }

    fn exec(self: *Interpreter, stmt: *const Stmt) RuntimeError!void {
        switch (stmt.*) {
            .function_decl => |*f| try self.visitFunctionStmt(f),
            .expr => |s| {
                _ = try self.eval(s.expression);
            },
            .while_decl => |w| try self.visitWhileStmt(w),
            .if_decl => |i| try self.visitIfStmt(i),
            .block => |b| try self.visitBlockStmt(b),
            .var_decl => |s| try self.visitVarStmt(s),
            .print => |s| try self.visitPrintStmt(s),
            .return_stmt => |r| try self.visitReturnStmt(r),
        }
    }

    fn execBlock(
        self: *Interpreter,
        stmts: []const *Stmt,
        env: *Environment,
    ) RuntimeError!void {
        const previous = self.env;
        self.env = env;
        defer self.env = previous;

        for (stmts) |stmt| {
            try self.exec(stmt);
        }
    }

    fn visitBlockStmt(self: *Interpreter, stmt: BlockStmt) RuntimeError!void {
        const env_ptr = try self.alloc.create(Environment);
        env_ptr.* = Environment.init(self.alloc, self.env);
        try self.execBlock(stmt.statements, env_ptr);
    }

    fn visitFunctionStmt(self: *Interpreter, stmt: *const FunctionStmt) RuntimeError!void {
        const function = UserFunction.init(stmt, self.env);

        const callable = self.alloc.create(LoxCallable) catch {
            return RuntimeError.OutOfMemory;
        };

        callable.* = LoxCallable{ .userFn = function };
        try self.env.define(stmt.name.lexeme, Value{ .callable = callable });
    }

    fn visitReturnStmt(self: *Interpreter, stmt: ReturnStmt) RuntimeError!void {
        if (stmt.value) |expr| {
            self.returnSignal = try self.eval(expr);
        } else {
            self.returnSignal = Value{ .nil = {} };
        }
        return RuntimeError.Return;
    }

    fn visitPrintStmt(self: *Interpreter, stmt: PrintStmt) RuntimeError!void {
        const v = try self.eval(stmt.expression);

        switch (v) {
            .string => |s| {
                const escaped = escapeString(self.alloc, s) catch {
                    return RuntimeError.TypeError;
                };
                self.writer.writeAll(escaped) catch {
                    return RuntimeError.TypeError;
                };
            },
            else => printValue(v, self.writer) catch {
                return RuntimeError.TypeError;
            },
        }
    }

    fn visitAssignExpr(self: *Interpreter, exprPtr: *Expr, expr: AssignExpr) RuntimeError!Value {
        const value = try self.eval(expr.value);
        if (self.locals.get(exprPtr)) |distance| {
            try self.env.assignAt(distance, expr.name.lexeme, value);
        } else {
            try self.globalEnv.assign(expr.name.lexeme, value);
        }
        return value;
    }

    fn visitIfStmt(self: *Interpreter, stmt: IFStmt) RuntimeError!void {
        const value = try self.eval(stmt.condition);

        if (self.isTruthy(value)) {
            try self.exec(stmt.thenBranch);
        } else if (stmt.elseBranch) |elseBranch| {
            try self.exec(elseBranch);
        }
    }

    fn visitVarStmt(self: *Interpreter, stmt: VarStmt) RuntimeError!void {
        var v: Value = Value.nil;
        if (stmt.initializer) |expr| {
            v = try self.eval(expr);
        }
        try self.env.define(stmt.name.lexeme, v);
    }

    fn visitWhileStmt(self: *Interpreter, stmt: WhileStmt) RuntimeError!void {
        while (self.isTruthy(try self.eval(stmt.condition))) {
            try self.exec(stmt.body);
        }
    }

    fn visitVariableExpr(self: *Interpreter, exprPrt: *Expr, expr: VariableExpr) RuntimeError!Value {
        return try self.lookUpVariable(expr.name, exprPrt);
    }

    fn lookUpVariable(self: *Interpreter, name: Token, expr: *Expr) RuntimeError!Value {
        if (self.locals.get(expr)) |distance| {
            return try self.env.getAt(distance, name.lexeme);
        } else {
            return try self.globalEnv.get(name.lexeme);
        }
    }

    fn visitLiteralExpr(_: *Interpreter, expr: LiteralExpr) RuntimeError!Value {
        return switch (expr.value) {
            .number => |n| .{ .number = n },
            .string => |s| .{ .string = s },
            .boolean => |b| .{ .boolean = b },
            .nil => Value.nil,
        };
    }

    fn visitLogicalExpr(self: *Interpreter, expr: LogicalExpr) RuntimeError!Value {
        const left = try self.eval(expr.left);
        if (expr.operator.type == .OR) {
            if (self.isTruthy(left)) return left;
        } else {
            if (!self.isTruthy(left)) return left;
        }
        return self.eval(expr.right);
    }

    fn visitUnaryExpr(self: *Interpreter, expr: UnaryExpr) RuntimeError!Value {
        const right = try self.eval(expr.right);
        return switch (expr.operator.type) {
            .MINUS => .{ .number = -try self.expectNumber(right) },
            .BANG => .{ .boolean = !self.isTruthy(right) },
            else => unreachable,
        };
    }

    fn visitBinaryExpr(self: *Interpreter, expr: BinaryExpr) RuntimeError!Value {
        const left = try self.eval(expr.left);
        const right = try self.eval(expr.right);

        return switch (expr.operator.type) {
            .PLUS => switch (left) {
                .number => |l| switch (right) {
                    .number => |r| .{ .number = l + r },
                    else => RuntimeError.TypeError,
                },
                .string => |l| switch (right) {
                    .string => |r| .{ .string = try self.concatStrings(l, r) },
                    else => RuntimeError.TypeError,
                },
                else => RuntimeError.TypeError,
            },
            .MINUS => .{
                .number = try self.expectNumber(left) - try self.expectNumber(right),
            },
            .STAR => .{
                .number = try self.expectNumber(left) * try self.expectNumber(right),
            },
            .SLASH => .{
                .number = try self.expectNumber(left) / try self.expectNumber(right),
            },
            .GREATER => .{
                .boolean = try self.expectNumber(left) > try self.expectNumber(right),
            },
            .GREATER_EQUAL => .{
                .boolean = try self.expectNumber(left) >= try self.expectNumber(right),
            },
            .LESS => .{
                .boolean = try self.expectNumber(left) < try self.expectNumber(right),
            },
            .LESS_EQUAL => .{
                .boolean = try self.expectNumber(left) <= try self.expectNumber(right),
            },
            .EQUAL_EQUAL => .{ .boolean = self.isEqual(left, right) },
            .BANG_EQUAL => .{ .boolean = !self.isEqual(left, right) },
            else => unreachable,
        };
    }

    fn visitCallExpr(self: *Interpreter, expr: CallExpr) RuntimeError!Value {
        const callee = try self.eval(expr.callee);
        var arguments = std.ArrayList(Value).initCapacity(self.alloc, 128) catch {
            return RuntimeError.OutOfMemory;
        };

        if (expr.arguments) |args| {
            for (args) |arg| {
                const v = try self.eval(arg);
                arguments.append(self.alloc, v) catch {
                    return RuntimeError.OutOfMemory;
                };
            }
        }

        return switch (callee) {
            .callable => |func| {
                var callable = func.*;

                if (arguments.items.len > callable.arity()) {
                    return RuntimeError.TooManyArguments;
                }
                if (arguments.items.len < callable.arity()) {
                    return RuntimeError.NotEnoughArguments;
                }

                const v = try callable.call(self, arguments.items);
                if (v) |value| {
                    return value;
                }
                return Value{ .nil = {} };
            },
            else => return RuntimeError.TypeError,
        };
    }

    fn isTruthy(_: *Interpreter, value: Value) bool {
        return switch (value) {
            .nil => false,
            .boolean => |b| b,
            else => true,
        };
    }

    pub fn resolve(self: *Interpreter, expr: *Expr, depth: usize) void {
        _ = self.locals.put(expr, depth) catch {};
    }

    fn concatStrings(
        self: *Interpreter,
        a: []const u8,
        b: []const u8,
    ) RuntimeError![]const u8 {
        const buf = self.alloc.alloc(u8, a.len + b.len) catch {
            return RuntimeError.OutOfMemory;
        };
        @memcpy(buf[0..a.len], a);
        @memcpy(buf[a.len..], b);
        return buf;
    }

    fn isEqual(_: *Interpreter, v1: Value, v2: Value) bool {
        return switch (v1) {
            .callable => {
                return false;
            },
            .nil => v2 == .nil,
            .number => |n1| switch (v2) {
                .number => |n2| n1 == n2,
                else => false,
            },
            .boolean => |b1| switch (v2) {
                .boolean => |b2| b1 == b2,
                else => false,
            },
            .string => |s1| switch (v2) {
                .string => |s2| std.mem.eql(u8, s1, s2),
                else => false,
            },
        };
    }

    fn expectNumber(_: *Interpreter, value: Value) RuntimeError!f64 {
        return switch (value) {
            .number => |n| n,
            else => RuntimeError.TypeError,
        };
    }

    fn clock_fn(_: *Interpreter, _: []const Value) RuntimeError!?Value {
        const ml: f64 = @floatFromInt(std.time.milliTimestamp());
        const t: f64 = ml / 1000.0;
        return Value{ .number = t };
    }

    fn runtimeError(
        _: *Interpreter,
        err: RuntimeError,
        message: ?[]const u8,
        token: ?Token,
    ) void {
        if (token) |tok| {
            if (message) |msg| {
                std.log.err(
                    "[line {d}] RuntimeError ({s}): {s}\n",
                    .{ tok.line, @errorName(err), msg },
                );
            } else {
                std.log.err(
                    "[line {d}] RuntimeError ({s})\n",
                    .{ tok.line, @errorName(err) },
                );
            }
        } else {
            if (message) |msg| {
                std.log.err(
                    "RuntimeError ({s}): {s}\n",
                    .{ @errorName(err), msg },
                );
            } else {
                std.log.err(
                    "RuntimeError ({s})\n",
                    .{@errorName(err)},
                );
            }
        }
    }
};
