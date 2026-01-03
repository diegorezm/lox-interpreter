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
    nil,
};

pub const Expr = union(enum) {
    Binary: BinaryExpr,
    Unary: UnaryExpr,
    Literal: LiteralExpr,
    Grouping: GroupingExpr,
    Variable: VariableExpr,
    Assign: AssignExpr,
    Logical: LogicalExpr,
};

pub const ExprStmt = struct {
    expression: *Expr,
};

pub const PrintStmt = struct {
    expression: *Expr,
};

pub const VarStmt = struct {
    name: Token,
    initializer: ?*Expr, // `var a;` is valid
};

pub const WhileStmt = struct {
    condition: *Expr,
    body: *Stmt,
};

pub const BlockStmt = struct {
    statements: []Stmt,
};

pub const IFStmt = struct {
    condition: *Expr,
    thenBranch: *Stmt,
    elseBranch: ?*Stmt,
};

pub const Stmt = union(enum) { expr: ExprStmt, print: PrintStmt, var_decl: VarStmt, block: BlockStmt, if_decl: IFStmt, while_decl: WhileStmt };

pub const Stmts = std.ArrayList(Stmt);

const ParserError = error{ ExpectedExpression, ExpectedRightParen, UnexpectedToken, OutOfMemory, InvalidAssignment };

pub const RuntimeError = error{ UndefinedVariable, InvalidOperands, DivisionByZero, OutputError, OutOfMemory, TypeError };
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

    pub fn parse(self: *Parser) !Stmts {
        var stmts: std.ArrayList(Stmt) = .empty;

        while (!self.isAtEnd()) {
            if (self.declaration()) |stmt| {
                try stmts.append(self.alloc, stmt.*);
            }
        }

        return stmts;
    }

    fn declaration(self: *Parser) ?*Stmt {
        const tokenTypes = [_]TokenType{.VAR};

        if (self.match(&tokenTypes)) {
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
        if (self.match(&[_]TokenType{TokenType.WHILE})) return try self.whileStatement();
        if (self.match(&[_]TokenType{TokenType.LEFT_BRACE})) {
            const b = try self.block();
            const stmt = self.alloc.create(Stmt) catch {
                return ParserError.OutOfMemory;
            };
            stmt.* = Stmt{ .block = BlockStmt{ .statements = b.items } };
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
            var stmts: Stmts = .empty;
            errdefer stmts.deinit(self.alloc);

            try stmts.append(self.alloc, body.*);

            const incStmt = try self.alloc.create(Stmt);
            incStmt.* = Stmt{
                .expr = ExprStmt{ .expression = inc },
            };
            try stmts.append(self.alloc, incStmt.*);

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
            var stmts: Stmts = .empty;
            errdefer stmts.deinit(self.alloc);

            try stmts.append(self.alloc, i.*);
            try stmts.append(self.alloc, body.*);

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

    fn block(self: *Parser) ParserError!std.ArrayList(Stmt) {
        var stmts: std.ArrayList(Stmt) = .empty;

        while (!self.check(TokenType.RIGHT_BRACE) and !self.isAtEnd()) {
            if (self.declaration()) |stmt| {
                stmts.append(self.alloc, stmt.*) catch {
                    return ParserError.OutOfMemory;
                };
            }
        }
        _ = try self.consume(TokenType.RIGHT_BRACE, "Expect '}' after block.");
        return stmts;
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
        name: Token,
        value: Value,
    ) RuntimeError!void {
        self.values.put(name.lexeme, value) catch {
            return RuntimeError.OutOfMemory;
        };
    }

    // Assigns a variable. This is different from `define` because here we are not creating any new
    // entries in the hashmap, only modifying entries that already exists.
    pub fn assign(
        self: *Environment,
        name: Token,
        value: Value,
    ) RuntimeError!void {
        if (self.values.contains(name.lexeme)) {
            self.values.put(name.lexeme, value) catch {
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
        name: Token,
    ) RuntimeError!Value {
        if (self.values.get(name.lexeme)) |value| {
            return value;
        }

        if (self.enclosing) |env| {
            return env.get(name);
        }

        return RuntimeError.UndefinedVariable;
    }
};

/// Executes the AST produced by the `Parser` and evaluates expressions.
pub const Interpreter = struct {
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    env: *Environment,

    pub fn init(alloc: std.mem.Allocator, writer: *std.io.Writer) !Interpreter {
        const env = try alloc.create(Environment);
        env.* = Environment.init(alloc, null);
        return .{ .alloc = alloc, .writer = writer, .env = env };
    }

    pub fn interpret(self: *Interpreter, stmts: *const Stmts) void {
        for (stmts.items) |stmt| {
            self.exec(&stmt) catch |err| {
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
            .Variable => |variable| try self.visitVariableExpr(variable),
            .Assign => |assign| try self.visitAssignExpr(assign),
            .Logical => |logical| try self.visitLogicalExpr(logical),
        };
    }

    fn exec(self: *Interpreter, stmt: *const Stmt) RuntimeError!void {
        switch (stmt.*) {
            .expr => |s| {
                _ = try self.eval(s.expression);
            },
            .while_decl => |w| try self.visitWhileStmt(w),
            .if_decl => |i| try self.visitIfStmt(i),
            .block => |b| try self.visitBlockStmt(b),
            .var_decl => |s| try self.visitVarStmt(s),
            .print => |s| try self.visitPrintStmt(s),
        }
    }

    fn execBlock(
        self: *Interpreter,
        stmts: []const Stmt,
        env: *Environment,
    ) RuntimeError!void {
        const previous = self.env;
        self.env = env;
        defer self.env = previous;

        for (stmts) |stmt| {
            try self.exec(&stmt);
        }
    }

    fn visitBlockStmt(self: *Interpreter, stmt: BlockStmt) RuntimeError!void {
        var localEnv = Environment.init(self.alloc, self.env);
        try self.execBlock(stmt.statements, &localEnv);
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

    fn visitAssignExpr(self: *Interpreter, expr: AssignExpr) RuntimeError!Value {
        const value = try self.eval(expr.value);
        try self.env.assign(expr.name, value);
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
        try self.env.define(stmt.name, v);
    }

    fn visitWhileStmt(self: *Interpreter, stmt: WhileStmt) RuntimeError!void {
        while (self.isTruthy(try self.eval(stmt.condition))) {
            try self.exec(stmt.body);
        }
    }

    fn visitVariableExpr(self: *Interpreter, expr: VariableExpr) RuntimeError!Value {
        return self.env.get(expr.name);
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
