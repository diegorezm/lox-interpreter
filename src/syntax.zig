const std = @import("std");

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

    // Turn this token into a json array.
    pub fn toString(self: Token, alloc: std.mem.Allocator) ![]const u8 {
        const fmt = std.json.fmt(self, .{ .whitespace = .indent_2 });

        var writer = std.Io.Writer.Allocating.init(alloc);
        try fmt.format(&writer.writer);

        const json_string = try writer.toOwnedSlice();
        return json_string;
    }
};

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

// Turn a slice into a keyword
pub fn keywordType(text: []const u8) TokenType {
    return keywords.get(text) orelse .IDENTIFIER;
}

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

pub const GroupingExpr = struct {
    expression: *Expr,
};

pub const Value = union(enum) {
    number: f64,
    boolean: bool,
    string: []const u8,
    nil,
};

pub fn valueToString(
    allocator: std.mem.Allocator,
    value: Value,
) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();

    try printValue(value, list.writer());
    return list.toOwnedSlice();
}

pub fn printValue(value: Value, writer: anytype) !void {
    switch (value) {
        .number => |n| try writer.print("{d}", .{n}),
        .boolean => |b| try writer.print("{}", .{b}),
        .string => |s| try writer.print("{s}", .{s}),
        .nil => try writer.writeAll("nil"),
    }
}

pub const Expr = union(enum) {
    Binary: BinaryExpr,
    Unary: UnaryExpr,
    Literal: LiteralExpr,
    Grouping: GroupingExpr,
};

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

const ParserError = error{
    ExpectedExpression,
    ExpectedRightParen,
    UnexpectedToken,
    OutOfMemory,
};

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

    pub fn parse(self: *Parser) !*Expr {
        return self.expression();
    }

    fn expression(self: *Parser) ParserError!*Expr {
        return try self.equality();
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

        const tokenTypes = [_]TokenType{ TokenType.GREATER, TokenType.GREATER_EQUAL, TokenType.LESS_EQUAL };

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
            return makeUnary(self.alloc, operator, right);
        }

        return try self.primary();
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

        if (self.match(&[_]TokenType{.LEFT_PAREN})) {
            const expr = try self.expression();
            _ = try self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.");
            return try makeGrouping(self.alloc, expr);
        }

        const p = self.peek();
        if (p == null) return ParserError.ExpectedExpression;

        reportParseError(p.?, "Expected expression.");
        return ParserError.ExpectedExpression;
    }

    fn sync(self: *Parser) void {
        self.advance();

        while (!self.isAtEnd()) {
            if (self.previous().type == .SEMICOLON) {
                return;
            }

            switch (self.peek().type) {
                .CLASS,
                .FUN,
                .VAR,
                .FOR,
                .IF,
                .WHILE,
                .PRINT,
                .RETURN,
                => return,

                else => {},
            }

            self.advance();
        }
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

    fn consume(self: *Parser, t: TokenType, message: []const u8) !Token {
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

pub const Interpreter = struct {
    alloc: std.mem.Allocator,
    pub fn init(alloc: std.mem.Allocator) Interpreter {
        return .{ .alloc = alloc };
    }

    pub fn interpret(self: *Interpreter, expr: *Expr) Value {
        return self.eval(expr);
    }

    fn eval(self: *Interpreter, expr: *Expr) Value {
        return switch (expr.*) {
            .Literal => |lit| self.literalExpr(lit),
            .Grouping => |grp| self.eval(grp.expression),
            .Binary => |binary| self.binaryExpr(binary),
            .Unary => |unary| self.unaryExpr(unary),
        };
    }

    fn literalExpr(self: *Interpreter, expr: LiteralExpr) Value {
        _ = self;
        return switch (expr.value) {
            .number => |n| Value{ .number = n },
            .string => |s| Value{ .string = s },
            .boolean => |b| Value{ .boolean = b },
            .nil => Value.nil,
        };
    }

    fn unaryExpr(self: *Interpreter, expr: UnaryExpr) Value {
        const right = self.eval(expr.right);
        switch (expr.operator.type) {
            .MINUS => {
                return Value{ .number = -self.expectNumber(right) };
            },
            .BANG => {
                return Value{ .boolean = !self.isTruthy(right) };
            },
            else => unreachable,
        }
    }

    fn binaryExpr(self: *Interpreter, expr: BinaryExpr) Value {
        const left = self.eval(expr.left);
        const right = self.eval(expr.right);

        return switch (expr.operator.type) {
            .PLUS => switch (left) {
                .number => |l| switch (right) {
                    .number => |r| .{ .number = l + r },
                    else => self.runtimeError("Operands must be two numbers or two strings."),
                },

                .string => |l| switch (right) {
                    .string => |r| {
                        const concat = self.concatStrings(l, r) catch {
                            self.runtimeError("Could not concataned thos strings brother.");
                            return "";
                        };
                        return .{ .string = concat };
                    },
                    else => self.runtimeError("Operands must be two numbers or two strings."),
                },
                else => self.runtimeError("Operands must be two numbers or two strings."),
            },

            .MINUS => {
                const l = self.expectNumber(left);
                const r = self.expectNumber(right);
                return .{ .number = l - r };
            },

            .STAR => {
                const l = self.expectNumber(left);
                const r = self.expectNumber(right);
                return .{ .number = l * r };
            },

            .SLASH => {
                const l = self.expectNumber(left);
                const r = self.expectNumber(right);
                return .{ .number = l / r };
            },

            .GREATER => {
                const l = self.expectNumber(left);
                const r = self.expectNumber(right);
                return .{ .boolean = l > r };
            },

            .GREATER_EQUAL => {
                const l = self.expectNumber(left);
                const r = self.expectNumber(right);
                return .{ .boolean = l >= r };
            },

            .LESS => {
                const l = self.expectNumber(left);
                const r = self.expectNumber(right);
                return .{ .boolean = l < r };
            },

            .LESS_EQUAL => {
                const l = self.expectNumber(left);
                const r = self.expectNumber(right);
                return .{ .boolean = l <= r };
            },

            .EQUAL_EQUAL => .{ .boolean = self.isEqual(left, right) },

            .BANG_EQUAL => .{ .boolean = !self.isEqual(left, right) },

            else => unreachable,
        };
    }

    fn isTruthy(_: *Interpreter, value: Value) bool {
        return switch (value) {
            .nil => false,
            .boolean => |b| b,
            else => true,
        };
    }
    fn concatStrings(
        self: *Interpreter,
        a: []const u8,
        b: []const u8,
    ) ![]const u8 {
        const buf = try self.alloc.alloc(u8, a.len + b.len);
        @memcpy(buf[0..a.len], a);
        @memcpy(buf[a.len..], b);
        return buf;
    }
    //
    // I know this is ugly, send an email to openAI if you have any complains
    fn isEqual(_: *Interpreter, v1: Value, v2: Value) bool {
        return switch (v1) {
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

    fn expectNumber(_: *Interpreter, value: Value) f64 {
        return switch (value) {
            .number => |n| n,
            else => {
                // I hate the idea of using panic here, will go back to this later (maybe never)
                @panic("Operand must be a number.");
            },
        };
    }
    fn runtimeError(self: *Interpreter, message: []const u8) noreturn {
        _ = self; // silence unused warning if not used yet
        std.debug.panic("Runtime error: {s}", .{message});
    }
};
